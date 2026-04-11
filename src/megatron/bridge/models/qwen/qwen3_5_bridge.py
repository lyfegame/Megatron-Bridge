# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Megatron Bridge for the **text backbone** of Qwen3.5 vision-language models.

Qwen3.5 models (0.8B, 2B, 4B, 9B, 27B) are VLMs with HF architecture
``Qwen3_5ForConditionalGeneration``.  The text backbone uses a hybrid
GatedDeltaNet (linear) + Gated Attention (full) architecture similar to
Qwen3-Next but with a **dense MLP** (no MoE) and separate linear-attention
projections (``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``
instead of the fused ``in_proj_qkvz`` / ``in_proj_ba`` used by Qwen3-Next).

HuggingFace weight prefix: ``model.language_model.layers.*`` (VLM nesting).
Vision encoder weights (``model.visual.*``) are ignored.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    MegatronParamMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.utils import remove_non_pickleables
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen_provider import Qwen35ModelProvider


# ---------------------------------------------------------------------------
# Qwen3.5 GDN linear-projection mapping (4 separate HF tensors)
# ---------------------------------------------------------------------------


def _merge_qwen35_gdn_linear(
    config: TransformerConfig,
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    tp_size: int = 1,
) -> torch.Tensor:
    """Merge Qwen3.5's 4 separate linear-attention projections into a single
    fused ``in_proj`` weight compatible with Megatron-Core's GatedDeltaNet.

    Qwen3.5 HF layout (all flat, row-major):
        qkv  -- shape (2*qk_dim + v_dim, hidden)  = [Q_all | K_all | V_all]
        z    -- shape (v_dim, hidden)               = Z_all
        b    -- shape (num_v_heads, hidden)          = B_all  (per-head scalar proj)
        a    -- shape (num_v_heads, hidden)          = A_all

    Megatron-Core expects a single ``in_proj`` in **per-qk-head-group** order,
    interleaved for TP:  [q, k, v, z, b, a] per group, repeated per TP shard.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads

    # 1. Split flat QKV
    Q, K, V = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # 2. Reshape into per-qk-head groups
    Q = Q.reshape(num_qk_heads, qk_head_dim, hidden_size)
    K = K.reshape(num_qk_heads, qk_head_dim, hidden_size)
    V = V.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    Z = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    B = b.reshape(num_qk_heads, v_per_group, hidden_size)
    A = a.reshape(num_qk_heads, v_per_group, hidden_size)

    # 3. Concatenate per group and reshape for TP
    Q, K, V, Z, B, A = [w.reshape(tp_size, -1, hidden_size) for w in [Q, K, V, Z, B, A]]
    in_proj = torch.cat([Q, K, V, Z, B, A], dim=1)
    in_proj = in_proj.reshape(-1, hidden_size)

    assert in_proj.numel() == qkv.numel() + z.numel() + b.numel() + a.numel(), (
        f"GDN linear merge size mismatch: "
        f"{qkv.numel()=} + {z.numel()=} + {b.numel()=} + {a.numel()=} != {in_proj.numel()=}"
    )
    return in_proj


def _split_qwen35_gdn_linear(
    config: TransformerConfig,
    in_proj: torch.Tensor,
    tp_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_merge_qwen35_gdn_linear`.

    Returns ``(qkv, z, b, a)`` in Qwen3.5's flat HF layout.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    num_qk_heads_local = num_qk_heads // tp_size
    num_v_heads_local = num_v_heads // tp_size
    v_per_group = num_v_heads // num_qk_heads
    qk_dim_local = qk_head_dim * num_qk_heads_local
    v_dim_local = v_head_dim * num_v_heads_local

    in_proj = in_proj.reshape(tp_size, -1, hidden_size)
    Q, K, V, Z, B, A = torch.split(
        in_proj,
        [qk_dim_local, qk_dim_local, v_dim_local, v_dim_local, num_v_heads_local, num_v_heads_local],
        dim=1,
    )

    # Reshape from per-group back to flat
    Q = Q.reshape(num_qk_heads, qk_head_dim, hidden_size)
    K = K.reshape(num_qk_heads, qk_head_dim, hidden_size)
    V = V.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    Z = Z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    B = B.reshape(num_qk_heads, v_per_group, hidden_size)
    A = A.reshape(num_qk_heads, v_per_group, hidden_size)

    qkv = torch.cat(
        [Q.reshape(-1, hidden_size), K.reshape(-1, hidden_size), V.reshape(-1, hidden_size)],
        dim=0,
    )
    z = Z.reshape(-1, hidden_size)
    b = B.reshape(-1, hidden_size)
    a = A.reshape(-1, hidden_size)

    return qkv, z, b, a


class Qwen35GDNLinearMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Maps Qwen3.5's 4 separate linear-attention projections to Megatron's
    single fused ``in_proj`` weight.

    HF side: ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``
    Megatron side: ``self_attention.in_proj.weight``
    """

    def __init__(self, megatron_param: str, qkv: str, z: str, b: str, a: str):
        super().__init__(megatron_param, {"qkv": qkv, "z": z, "b": b, "a": a})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            merged = _merge_qwen35_gdn_linear(
                config,
                hf_weights["qkv"],
                hf_weights["z"],
                hf_weights["b"],
                hf_weights["a"],
                tp_size=self.tp_size,
            )
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config)

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed = next(iter(packed_dict.values()))
        qkv, z, b, a = _split_qwen35_gdn_linear(config, packed, tp_size=self.tp_size)

        return {
            self.hf_param["qkv"]: qkv,
            self.hf_param["z"]: z,
            self.hf_param["b"]: b,
            self.hf_param["a"]: a,
        }

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["qkv"],
            resolved_hf_param["z"],
            resolved_hf_param["b"],
            resolved_hf_param["a"],
        )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

LM = "model.language_model"


@MegatronModelBridge.register_bridge(
    source="Qwen3_5ForConditionalGeneration",
    target=GPTModel,
    model_type="qwen3_5_text",
)
class Qwen35Bridge(MegatronModelBridge):
    """Megatron Bridge for the **text backbone** of Qwen3.5 VLMs.

    Supports dense models at 0.8B, 2B, 4B, 9B, 27B.
    Vision encoder weights (``model.visual.*``) are ignored.

    Example::

        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3.5-9B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen35ModelProvider:
        hf_config = hf_pretrained.config
        # Qwen3.5 is a VLM; the language model config is nested.
        tc = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

        rope_params = getattr(tc, "rope_parameters", {})
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta", 10_000_000)
            partial_rotary = rope_params.get("partial_rotary_factor", 0.25)
        else:
            rope_theta = getattr(rope_params, "rope_theta", 10_000_000)
            partial_rotary = getattr(rope_params, "partial_rotary_factor", 0.25)

        provider = Qwen35ModelProvider(
            num_layers=tc.num_hidden_layers,
            hidden_size=tc.hidden_size,
            ffn_hidden_size=tc.intermediate_size,
            num_attention_heads=tc.num_attention_heads,
            num_query_groups=tc.num_key_value_heads,
            init_method_std=tc.initializer_range,
            layernorm_epsilon=tc.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(tc.vocab_size),
            rotary_base=rope_theta,
            share_embeddings_and_output_weights=getattr(tc, "tie_word_embeddings", False),
            vocab_size=tc.vocab_size,
            seq_length=tc.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            qk_layernorm=True,
            kv_channels=tc.head_dim,
            attention_output_gate=getattr(tc, "attn_output_gate", True),
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=tc.full_attention_interval,
            rotary_percent=partial_rotary,
            linear_conv_kernel_dim=tc.linear_conv_kernel_dim,
            linear_key_head_dim=tc.linear_key_head_dim,
            linear_value_head_dim=tc.linear_value_head_dim,
            linear_num_key_heads=tc.linear_num_key_heads,
            linear_num_value_heads=tc.linear_num_value_heads,
            mtp_num_layers=getattr(tc, "mtp_num_hidden_layers", None),
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # -- Embedding & output --
            "embedding.word_embeddings.weight": f"{LM}.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": f"{LM}.norm.weight",
            # -- Full-attention layers --
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": f"{LM}.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": f"{LM}.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": f"{LM}.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": f"{LM}.layers.*.self_attn.o_proj.weight",
            # -- Linear-attention layers (GatedDeltaNet) --
            "decoder.layers.*.self_attention.in_proj.layer_norm_weight": f"{LM}.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.out_proj.weight": f"{LM}.layers.*.linear_attn.out_proj.weight",
            "decoder.layers.*.self_attention.A_log": f"{LM}.layers.*.linear_attn.A_log",
            "decoder.layers.*.self_attention.dt_bias": f"{LM}.layers.*.linear_attn.dt_bias",
            "decoder.layers.*.self_attention.out_norm.weight": f"{LM}.layers.*.linear_attn.norm.weight",
            # -- Dense MLP (all layers) --
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": f"{LM}.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": f"{LM}.layers.*.mlp.down_proj.weight",
            # -- MTP projection and norms --
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            # -- MTP attention --
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
            # -- MTP dense MLP --
            "mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.transformer_layer.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
        }

        mapping_list: List[MegatronParamMapping] = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        AutoMapping.register_module_type("GatedDeltaNet", "column")

        mapping_list.extend(
            [
                # -- Full-attention QKV --
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q=f"{LM}.layers.*.self_attn.q_proj.weight",
                    k=f"{LM}.layers.*.self_attn.k_proj.weight",
                    v=f"{LM}.layers.*.self_attn.v_proj.weight",
                ),
                # -- Linear-attention in_proj (4 separate HF tensors) --
                Qwen35GDNLinearMapping(
                    megatron_param="decoder.layers.*.self_attention.in_proj.weight",
                    qkv=f"{LM}.layers.*.linear_attn.in_proj_qkv.weight",
                    z=f"{LM}.layers.*.linear_attn.in_proj_z.weight",
                    b=f"{LM}.layers.*.linear_attn.in_proj_b.weight",
                    a=f"{LM}.layers.*.linear_attn.in_proj_a.weight",
                ),
                # -- Linear-attention conv1d --
                GDNConv1dMapping(
                    megatron_param="decoder.layers.*.self_attention.conv1d.weight",
                    hf_param=f"{LM}.layers.*.linear_attn.conv1d.weight",
                ),
                # -- Dense MLP (gated: gate + up -> fc1) --
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate=f"{LM}.layers.*.mlp.gate_proj.weight",
                    up=f"{LM}.layers.*.mlp.up_proj.weight",
                ),
                # -- MTP QKV --
                QKVMapping(
                    megatron_param="mtp.layers.*.transformer_layer.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                # -- MTP dense MLP --
                GatedMLPMapping(
                    megatron_param="mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.gate_proj.weight",
                    up="mtp.layers.0.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
