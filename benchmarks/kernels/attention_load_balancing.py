import argparse
import random
import time
import math
import torch
from vllm._C import ops
from typing import List

NUM_BLOCKS = 8192
PARTITION_SIZE = 512


class TooManyBlocks(Exception):
    pass


class KernelVersion:
    ORIGINAL = 0
    TARGET = 1
    TENSOR_CORE = 2


@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    context_lens: List[int],  # [num_seqs]
    query_lens: List[int],  # [num_seqs]
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:

    # Set random seeds
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))

    sum_query_lens = sum(query_lens)  # Query is jagged tensor of above shape
    query = torch.empty(
        sum_query_lens, num_query_heads, head_size, dtype=dtype, device="cuda"
    )

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )

    # alibi slope settings
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float, device="cuda")

    # Tensorize the context_lens and query_lens.
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
    max_context_len = context_lens.max().item()
    query_lens = torch.tensor(query_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []  # For original kernel
    total_num_blocks_needed = max_num_blocks_per_seq * num_seqs
    if total_num_blocks_needed > NUM_BLOCKS:
        raise TooManyBlocks(
            f"Too many blocks needed: {total_num_blocks_needed} > {NUM_BLOCKS}"
        )
    bt_idx = 0
    for seq_idx in range(num_seqs):  # (hj) : Valid in target context too!
        # Sequence 사이에는 공유하는 K/V 없다고 가정
        block_table = [i for i in range(bt_idx, bt_idx + max_num_blocks_per_seq)]
        bt_idx += max_num_blocks_per_seq
        # Align to up to block_size
        for _ in range(query_lens[seq_idx]):
            block_tables.append(block_table)

    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device="cuda")
    key_cache.uniform_(-scale, scale)

    value_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device="cuda")
    value_cache.uniform_(-scale, scale)

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)

    # (hj) Another output to validate our stuff
    validation_output = torch.empty_like(query)

    if version == "v2":
        num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE

        # (hj) : num_seqs -> sum_query_lens
        tmp_output = torch.empty(
            size=(sum_query_lens, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )

        # (hj) : num_seqs -> sum_query_lens
        exp_sums = torch.empty(
            size=(sum_query_lens, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    def run_benchmark(target: KernelVersion, num_iters: int = 100) -> float:

        def run():
            if target == KernelVersion.TARGET:
                ops.paged_attention_v2_target(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    query_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )
            elif target == KernelVersion.ORIGINAL:
                ops.paged_attention_v2(
                    validation_output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )
            else:
                ops.paged_attention_v2_target_tensor_core(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    query_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )

        # Warmup phase
        for _ in range(3):
            run()

        # Benchmark phase
        torch.cuda.synchronize()
        start_time = time.perf_counter_ns()
        for _ in range(num_iters):
            run()
        torch.cuda.synchronize()
        end_time = time.perf_counter_ns()

        elasped_time_ns = (end_time - start_time) / num_iters
        elasped_time_ms = elasped_time_ns / 1e6

        return elasped_time_ms

    torch.cuda.synchronize()
    target_latency = run_benchmark(target=KernelVersion.TARGET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel."
    )
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-size", type=int, choices=[64, 80, 96, 112, 128, 256], default=128
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--query-len", type=int, default=-1)  # -1 is random
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }

    # Variables to alter for each experiment

    MAX_QUERY_LEN = 16

    # -> num_seqs : Does saturating the GPU scheduler with tasks reduce the imbalance problem?
    # -> sum_context_lens : Similar intuition to above

    # Phase 1: Test with uniform distribution
    def gen_uniform_query_lens(num_seqs: int, query_len: int) -> List[int]:
        query_lens = [query_len] * num_seqs
        return query_lens

    # Phase 2: Test with normal distribution
    def gen_normal_query_lens(num_seqs: int, query_len: int, std: float) -> List[int]:
        mean = query_len
        query_lens = [
            max(min(math.ceil(random.normalvariate(mean, std)), MAX_QUERY_LEN), 0)
            for _ in range(num_seqs)
        ]
        return query_lens

    # Phase 3: Test with U-like distribution
    def gen_U_query_lens(num_seqs: int, query_len: int, std: float) -> List[int]:
        min_query_len = 1
        max_query_len = (
            2 * query_len - min_query_len
        )  # this makes average query length to be query_len
        query_lens_min = [min_query_len] * (num_seqs // 2)
        query_lens_max = [max_query_len] * (num_seqs // 2)
        query_lens = query_lens_min + query_lens_max
        return query_lens

    # simple test :) TODO: tie to function and run before test

    num_seqs = 100
    avg_query_len = 8
    std = 2

    normal_query_lens = gen_normal_query_lens(num_seqs, avg_query_len, std)
    U_query_lens = gen_U_query_lens(num_seqs, avg_query_len, std)

    # Draw diagram
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].hist(
        normal_query_lens,
        bins=np.arange(0, MAX_QUERY_LEN + 1, 1),
        color="blue",
        alpha=0.5,
        label="Normal",
    )
    ax[0].set_title("Normal Distribution")
    ax[0].set_xlabel("Query Length")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()

    ax[1].hist(
        U_query_lens, bins=np.arange(0, MAX_QUERY_LEN + 1, 1), color="red", alpha=0.5, label="U-like"
    )
    ax[1].set_title("U-like Distribution")
    ax[1].set_xlabel("Query Length")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    # Save as test.jpg
    plt.savefig("distribution.jpg")
    
    
    
    

    # main(
    #     version=args.version,
    #     num_seqs=args.batch_size,
    #     context_lens=context_lens,
    #     query_lens=query_lens,
    #     num_query_heads=args.num_query_heads,
    #     num_kv_heads=args.num_kv_heads,
    #     head_size=args.head_size,
    #     block_size=args.block_size,
    #     use_alibi=args.use_alibi,
    #     dtype=dtype_to_torch_dtype[args.dtype],
    #     seed=args.seed,
    # )
