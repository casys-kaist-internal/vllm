import argparse
import random
import time
import math
import torch
from vllm._C import ops
from typing import List

NUM_BLOCKS = 50000
PARTITION_SIZE = 512

import numpy as np


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
            for context_len in range(window_size, 1024, 4):
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
            if window_size == 1:
                x = [r[0] for r in orig_results]
                y = [r[1] for r in orig_results]
                plt.plot(
                    x,
                    y,
                    label=f"Naive kernel with Window Size : {window_size}",
                    linestyle="dashed",
                    color=colours[window_size],
                )

        ######
        plt.legend()

        # X-axis label
        plt.xlabel("Context Length")

        # Y-axis label
        plt.ylabel("Latency (ms)")

        # x : show 0 ~ 2048
        # y : show 0 ~ 0.4
        plt.xlim(0, 1024)
        plt.ylim(0, 1)

        plt.savefig(f"attention_kernel_analysis.jpg")

    @torch.inference_mode()
    def run_batch_size_analysis():

        import matplotlib.pyplot as plt

        # Set plot size
        plt.figure(figsize=(10, 10))

        colours = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

        context_len = 512

        for window_size in range(8, 0, -1):
            print(f"Running with window size: {window_size}")
            target_results = []
            orig_results = []
            # for context_len in range(query_len, 1024, 4):
            for num_seqs in range(1, 32, 1):
                print(f"Running with bs length: {num_seqs}")
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
                target_results.append((num_seqs, target))
                orig_results.append((num_seqs, orig))

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

            ###### Draw plot per window_size
            plt.legend()

            # X-axis label
            plt.xlabel("Batch Size")

            # Y-axis label
            plt.ylabel("Latency (ms)")

            plt.savefig(f"batch_size_analysis_window_{window_size}.jpg")
            # reset plot
            plt.clf()
            print()

    @torch.inference_mode()
    def run_cl_var_analysis():
        import matplotlib.pyplot as plt

        # Set plot size
        plt.figure(figsize=(10, 10))

        colours = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

        num_seqs = (
            128  # NOTE: We use large value to get a clear picture of the distribution
        )

        avg_context_len = 1024

        variances = [i for i in range(4, 1024, 32)]
        for variance in variances:
            context_lens = []
            for _ in range(num_seqs):
                cl = int(random.gauss(avg_context_len, variance))
                cl = max(8, cl)
                cl = min(2048, cl)
                context_lens.append(cl)

            # Plot and save context-lens
            plt.hist(context_lens, bins=100)
            plt.xlabel("Context Length")
            plt.ylabel("Frequency")
            plt.xlim(0, 2200)
            plt.title(f"Context Length Distribution with Variance: {variance}")
            plt.savefig(f"context_len_dist_variance_{variance}.jpg")
            plt.clf()

        for window_size in range(8, 0, -1):
            print(f"Running with window size: {window_size}")
            target_results = []
            orig_results = []
            # for context_len in range(query_len, 1024, 4):
            for variance in variances:
                context_lens = []
                for _ in range(num_seqs):
                    cl = int(random.gauss(avg_context_len, variance))
                    cl = max(8, cl)
                    cl = min(2048, cl)
                    context_lens.append(cl)

                context_lens = gen_context_len(context_lens, [window_size] * num_seqs)

                print(f"Running var: {variance}")
                query_lens = [window_size] * num_seqs

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
                target_results.append((variance, target))
                orig_results.append((variance, orig))

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

        ###### Draw plot per window_size
        plt.legend()

        # X-axis label
        plt.xlabel("Variance")

        # Y-axis label
        plt.ylabel("Latency (ms)")

        plt.savefig(f"variance_analysis.jpg")
        # reset plot
        plt.clf()
        print()

    @torch.inference_mode()
    def run_det_exp_bi1_bi2():

        def make_bimodal_dist(
            mean_1, mean_2, std_1, std_2, ratio, n, min_val=8, max_val=9999999
        ):
            c1 = int(n * ratio)
            c2 = max(int(n * (1 - ratio)), 2)
            # Make n * ratio normal distribution
            context_lens = []
            for _ in range(c1):
                cl = int(random.gauss(mean_1, std_1))
                cl = max(min_val, cl)
                cl = min(max_val, cl)
                context_lens.append(cl)

            for _ in range(c2):
                cl = int(random.gauss(mean_2, std_2))
                cl = max(min_val, cl)
                cl = min(max_val, cl)
                context_lens.append(cl)
            return context_lens

        num_seqs = 256
        import matplotlib.pyplot as plt

        # Make 4 plots
        # 1. Deterministic context length
        # 2. Exponential context length
        # 3. Bi-modal 1 context length
        # 4. Bi-modal 2 context length

        # Miniplot for each distribution
        avg_context_len = 512
        plt.figure(figsize=(10, 10))

        # Exponential distribution
        exp_context_lens = []
        for _ in range(num_seqs):
            cl = int(np.random.default_rng().exponential(avg_context_len))
            cl = max(8, cl)
            cl = min(99999, cl)
            exp_context_lens.append(cl)

        # Bi-modal 1
        bi1_context_lens = make_bimodal_dist(
            mean_1=avg_context_len // 2,
            mean_2=avg_context_len * 5.5,
            std_1=avg_context_len // 4,
            std_2=avg_context_len // 4,
            ratio=0.9,
            n=num_seqs,
        )

        def save_fig(context_lens, title):
            plt.hist(context_lens, bins=num_seqs)
            plt.xlabel("Context Length")
            plt.ylabel("Frequency")
            plt.xlim(0, 4096)
            plt.title(title)
            plt.savefig(f"{title}.jpg")
            plt.clf()

        save_fig(exp_context_lens, "Exponential Context Length Distribution")
        save_fig(bi1_context_lens, "Bi-modal 1 Context Length Distribution")

        ##########

        plt.figure(figsize=(10, 10))
        # 4 subplots
        fig, axs = plt.subplots(3, 1)

        axes_map = {
            "deterministic": axs[0],
            "exponential": axs[1],
            "bi1": axs[2],
        }

        colours = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]

        for window_size in [8, 4, 1]:
            print(f"Running with window size: {window_size}")
            target_results = {"deterministic": [], "exponential": [], "bi1": []}
            orig_results = {"deterministic": [], "exponential": [], "bi1": []}

            for avg_context_len in range(8, 1024, 64):
                query_lens = [window_size] * num_seqs

                # Deterministic
                deterministic_context_lens = gen_context_len(
                    [avg_context_len] * num_seqs, query_lens
                )
                # Exponential distribution
                exp_context_lens = []
                for _ in range(num_seqs):
                    cl = int(np.random.default_rng().exponential(avg_context_len))
                    cl = max(8, cl)
                    cl = min(2048, cl)
                    exp_context_lens.append(cl)

                exp_context_lens = gen_context_len(exp_context_lens, query_lens)

                # Bi-modal 1
                bi1_context_lens = make_bimodal_dist(
                    mean_1=avg_context_len // 2,
                    mean_2=avg_context_len * 5.5,
                    std_1=avg_context_len // 4,
                    std_2=avg_context_len // 4,
                    ratio=0.9,
                    n=num_seqs,
                )
                bi1_context_lens = gen_context_len(bi1_context_lens, query_lens)

                contexts = {
                    "deterministic": deterministic_context_lens,
                    "exponential": exp_context_lens,
                    "bi1": bi1_context_lens,
                }

                for dist_name, context_lens in contexts.items():
                    print(f"Running with context distribution: {dist_name}")

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
                    target_results[dist_name].append((avg_context_len, target))
                    orig_results[dist_name].append((avg_context_len, orig))

            x = [r[0] for r in target_results]
            y = [r[1] for r in target_results]

            for dist_name, ax in axes_map.items():
                ax.plot(
                    x,
                    y,
                    label=f"Modified kernel with Window Size : {window_size}",
                    color=colours[window_size],
                )

                # Draw line plot,
                if window_size == 1:
                    x = [r[0] for r in orig_results]
                    y = [r[1] for r in orig_results]
                    plt.plot(
                        x,
                        y,
                        label=f"Naive kernel with Window Size : {window_size}",
                        linestyle="dashed",
                        color=colours[window_size],
                    )

        ######
        plt.legend()

        # X-axis label
        plt.xlabel("Context Length")

        # Y-axis label
        plt.ylabel("Latency (ms)")

        # x : show 0 ~ 2048
        # y : show 0 ~ 0.4

        # Name subplots
        for dist_name, ax in axes_map.items():
            ax.set_title(f"{dist_name.capitalize()} Context Length Distribution")
            ax.set_xlabel("Context Length")
            ax.set_ylabel("Latency (ms)")
            ax.legend()

        plt.savefig(f"context_with_diff_distributions.jpg")

    # run_test()
    # run_batch_size_analysis()

    # run_test()

    # run_cl_var_analysis()

    run_det_exp_bi1_bi2()
