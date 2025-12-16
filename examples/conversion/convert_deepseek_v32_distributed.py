#!/usr/bin/env python3
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
DeepSeek V3.2 Distributed Checkpoint Conversion

This script converts DeepSeek V3.2 checkpoints from HuggingFace format to Megatron
format WITH the target tensor parallelism (TP) and expert parallelism (EP) sharding.

WHY THIS SCRIPT EXISTS:
The standard `convert_deepseek_v32.py` runs single-process and saves at TP=1.
When loading at TP=8, the checkpoint requires expensive resharding that can OOM.
This script runs distributed, so each rank loads its shard and saves directly
at the target parallelism - no resharding needed at load time.

Usage:
    # Convert with TP=8, EP=2 (requires 16 GPUs across 2 nodes)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=10.0.0.17 --master_port=29500 \\
        convert_deepseek_v32_distributed.py \\
        --hf-path /path/to/DeepSeek-V3.2-fp8 \\
        --megatron-path /path/to/output \\
        --tp 8 --ep 2

    # Convert with TP=8 only (requires 8 GPUs on 1 node)
    torchrun --nproc_per_node=8 \\
        convert_deepseek_v32_distributed.py \\
        --hf-path /path/to/DeepSeek-V3.2-fp8 \\
        --megatron-path /path/to/output \\
        --tp 8 --ep 1

Memory Requirements:
    - Each GPU needs ~18GB for model shard + loading overhead
    - CPU memory: ~100GB per node for HF checkpoint loading
    - Disk: Output checkpoint will be ~1.3TB (BF16)

Note:
    The FP8 checkpoint will be automatically dequantized to BF16 during conversion
    since FP8 is only supported for inference, not training.
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s",
)


class RankLogFilter(logging.Filter):
    """Add rank to log records."""
    def filter(self, record):
        record.rank = dist.get_rank() if dist.is_initialized() else 0
        return True


logger = logging.getLogger(__name__)
logger.addFilter(RankLogFilter())


def setup_distributed():
    """Initialize distributed environment from torchrun."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(hours=2)  # Long timeout for large model loading
        )

    return rank, local_rank, world_size


def initialize_megatron_parallel(tp_size: int, ep_size: int, pp_size: int = 1):
    """Initialize Megatron parallel state with target parallelism."""
    from megatron.core import parallel_state

    world_size = dist.get_world_size()
    expected_world_size = tp_size * ep_size * pp_size

    if world_size != expected_world_size:
        raise ValueError(
            f"World size {world_size} doesn't match TP={tp_size} x EP={ep_size} x PP={pp_size} = {expected_world_size}"
        )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )

    if dist.get_rank() == 0:
        logger.info(f"Initialized Megatron parallel state:")
        logger.info(f"  Tensor Parallel (TP): {tp_size}")
        logger.info(f"  Expert Parallel (EP): {ep_size}")
        logger.info(f"  Pipeline Parallel (PP): {pp_size}")
        logger.info(f"  World Size: {world_size}")


def convert_distributed(
    hf_path: str,
    megatron_path: str,
    tp_size: int,
    ep_size: int,
    output_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> None:
    """
    Convert DeepSeek V3.2 checkpoint with distributed sharding.

    Each rank:
    1. Loads its shard of HF weights (via Bridge's hf_to_megatron)
    2. Saves its shard to the output checkpoint

    Args:
        hf_path: Path to the HuggingFace checkpoint
        megatron_path: Output directory for Megatron checkpoint
        tp_size: Tensor parallel size
        ep_size: Expert parallel size
        output_dtype: Target dtype for weights
        trust_remote_code: Whether to trust remote code from HuggingFace
    """
    from megatron.bridge import AutoBridge

    rank = dist.get_rank()

    if rank == 0:
        logger.info(f"Converting DeepSeek V3.2: {hf_path} -> {megatron_path}")
        logger.info(f"Target parallelism: TP={tp_size}, EP={ep_size}")
        logger.info(f"Output dtype: {output_dtype}")

    # Validate input path on rank 0
    if rank == 0:
        hf_path_obj = Path(hf_path)
        if not hf_path_obj.exists():
            raise FileNotFoundError(f"HuggingFace checkpoint not found: {hf_path}")

        config_path = hf_path_obj / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)

            if config.get("index_topk"):
                logger.info(f"Lightning Indexer config:")
                logger.info(f"  - index_n_heads: {config.get('index_n_heads', 'N/A')}")
                logger.info(f"  - index_head_dim: {config.get('index_head_dim', 'N/A')}")
                logger.info(f"  - index_topk: {config.get('index_topk', 'N/A')}")

            torch_dtype = config.get("torch_dtype", "")
            if "float8" in torch_dtype.lower():
                logger.info("Detected FP8 checkpoint - will auto-dequantize to BF16")

    dist.barrier()

    # Prepare torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(output_dtype, torch.bfloat16)

    if rank == 0:
        logger.info("Loading HuggingFace checkpoint and creating Megatron model...")
        logger.info("This may take a while for large models like DeepSeek V3.2 (685B)")

    # Create bridge from HF checkpoint
    # This loads the HF config and sets up weight mappings
    bridge = AutoBridge.from_hf_pretrained(
        hf_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if rank == 0:
        logger.info("Creating Megatron model with target parallelism...")

    # Create Megatron model - this creates the model structure
    # with proper TP/EP sharding based on the initialized parallel state
    megatron_model = bridge.to_megatron_model(
        load_weights=True,  # Load weights during model creation
        wrap_with_ddp=False,
        use_cpu_initialization=False,  # Use GPU for faster init
    )

    dist.barrier()

    if rank == 0:
        logger.info("Model created and weights loaded. Saving checkpoint...")

    # Save the checkpoint - each rank saves its shards
    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_path,
    )

    dist.barrier()

    if rank == 0:
        logger.info(f"Successfully converted checkpoint to: {megatron_path}")

        # Verify output
        output_path = Path(megatron_path)
        if output_path.exists():
            logger.info("Output checkpoint structure:")
            for item in sorted(output_path.iterdir()):
                if item.is_dir():
                    logger.info(f"  [DIR] {item.name}/")
                else:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    logger.info(f"  [FILE] {item.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek V3.2 checkpoint with distributed sharding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Path to HuggingFace checkpoint (local path)",
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Output directory for Megatron checkpoint",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=2,
        help="Expert parallel size (default: 2)",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Target dtype for weights (default: bfloat16)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Don't trust remote code from HuggingFace",
    )

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Starting distributed conversion with {world_size} GPUs")

    # Initialize Megatron parallel state
    initialize_megatron_parallel(args.tp, args.ep)

    # Run conversion
    try:
        convert_distributed(
            hf_path=args.hf_path,
            megatron_path=args.megatron_path,
            tp_size=args.tp,
            ep_size=args.ep,
            output_dtype=args.output_dtype,
            trust_remote_code=not args.no_trust_remote_code,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
