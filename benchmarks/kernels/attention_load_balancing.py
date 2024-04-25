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


def gen_context_len(max_context_len_per_query: List[int], query_lens: List[int]):
    res = []
    for query_len, c_max in zip(query_lens, max_context_len_per_query):
        for i in range(query_len):
            res.append(c_max - (query_len - i - 1))
    return res


def main_runner(
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
) -> float:

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

    tmp_output_target = torch.empty_like(tmp_output)
    exp_sums_target = torch.empty_like(exp_sums)
    max_logits_target = torch.empty_like(max_logits)

    def run_benchmark(target: KernelVersion, num_iters: int) -> float:
        def run():
            if target == KernelVersion.TARGET:
                ops.paged_attention_v2_target(
                    output,
                    exp_sums_target,
                    max_logits_target,
                    tmp_output_target,
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
                    exp_sums_target,
                    max_logits_target,
                    tmp_output_target,
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
    original_latency = run_benchmark(target=KernelVersion.ORIGINAL, num_iters=100)
    target_latency = run_benchmark(target=KernelVersion.TARGET, num_iters=100)
    print(f"Target latency: {target_latency:.2f} ms")
    print(f"Original latency: {original_latency:.2f} ms")
    return original_latency, target_latency


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
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=32)
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

    @torch.inference_mode()
    def run_test():
        num_seqs = 32
        import matplotlib.pyplot as plt

        # Set plot size
        plt.figure(figsize=(10, 10))

        colours = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

        for window_size in range(8, 0, -1):
            print(f"Running with window size: {window_size}")
            target_results = []
            orig_results = []
            for context_len in range(32, 513, 32):
                print(f"Running with context length: {context_len}")
                query_lens = [window_size] * num_seqs

                context_lens = gen_context_len([context_len] * num_seqs, query_lens)

                orig, target = main_runner(
                    num_seqs=num_seqs,
                    context_lens=context_lens,
                    query_lens=query_lens,
                    num_query_heads=args.num_query_heads,
                    num_kv_heads=args.num_kv_heads,
                    head_size=args.head_size,
                    block_size=args.block_size,
                    use_alibi=args.use_alibi,
                    dtype=dtype_to_torch_dtype[args.dtype],
                    seed=args.seed,
                )
                target_results.append((context_len, target))
                orig_results.append((context_len, orig))

            # Draw line plot,
            x = [r[0] for r in target_results]
            y = [r[1] for r in target_results]
            plt.plot(
                x,
                y,
                label=f"Modified kernel with Window Size : {window_size}",
                color=colours[window_size],
            )

            # Draw line plot,
            x = [r[0] for r in orig_results]
            y = [r[1] for r in orig_results]
            plt.plot(
                x,
                y,
                label=f"Naive kernel with Window Size : {window_size}",
                linestyle="dashed",
                color=colours[window_size],
            )

            # Save as image

        # Do reference plot
        # window_size = 1
        # results = []
        # orig_results = []
        # for context_len in range(32, 512, 32):
        #     print(f"Running with context length: {context_len}")
        #     query_lens = [window_size] * num_seqs

        #     context_lens = gen_context_len([context_len] * num_seqs, query_lens)

        #     orig, target = main_runner(
        #         num_seqs=num_seqs,
        #         context_lens=context_lens,
        #         query_lens=query_lens,
        #         num_query_heads=args.num_query_heads,
        #         num_kv_heads=args.num_kv_heads,
        #         head_size=args.head_size,
        #         block_size=args.block_size,
        #         use_alibi=args.use_alibi,
        #         dtype=dtype_to_torch_dtype[args.dtype],
        #         seed=args.seed,
        #     )
        #     results.append((context_len, target))
        #     orig_results.append((context_len, orig))

        # # Draw line plot,
        # x = [r[0] for r in results]
        # y = [r[1] for r in results]

        # # Plot with 2px, dashed line
        # plt.plot(
        #     x,
        #     y,
        #     label=f"Decoder Original algorithm : {window_size}",
        #     linestyle="dashed",
        #     linewidth=2,
        # )

        ######
        plt.legend()

        # X-axis label
        plt.xlabel("Context Length")

        # Y-axis label
        plt.ylabel("Latency (ms)")

        plt.savefig(f"attention_kernel_analysis.jpg")

    run_test()
