import argparse
import random
import time

import torch

from vllm._C import ops

<<<<<<< HEAD
NUM_BLOCKS = 1024
PARTITION_SIZE = 512
=======
from typing import List

NUM_BLOCKS = 1024
PARTITION_SIZE = 8
>>>>>>> tmp


@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
<<<<<<< HEAD
    context_len: int,
=======
    context_lens: List[int],  # [num_seqs]
    query_lens: List[int],  # [num_seqs]
>>>>>>> tmp
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
<<<<<<< HEAD
    do_profile: bool,
=======
>>>>>>> tmp
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

<<<<<<< HEAD
    query_len = 8 # for testing only

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
=======
    scale = float(1.0 / (head_size**0.5))

    # (hj) Query format is [num_seqs x query_lens, num_query_heads, head_size]

    sum_query_lens = sum(query_lens)  # Query is jagged tensor of above shape

    query = torch.empty(
        sum_query_lens, num_query_heads, head_size, dtype=dtype, device="cuda"
    )
>>>>>>> tmp
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
<<<<<<< HEAD
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [context_len for _ in range(num_seqs)]
    query_lens = [context_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
=======
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float, device="cuda")

    # (hj) Changed as we are passing context_lens as input
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
    max_context_len = context_lens.max().item()
    
    query_lens = torch.tensor(query_lens, dtype=torch.int, device="cuda")
>>>>>>> tmp

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
<<<<<<< HEAD
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
=======
    block_tables_validation = []
    for seq_idx in range(num_seqs):  # (hj) : Valid in target context too!
        block_table = [
            random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
        # For original kernel, all sequences have same block tables
        for _ in range(query_lens[seq_idx]):
            block_tables_validation.append(block_table)

    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    block_tables_validation = torch.tensor(block_tables_validation, dtype=torch.int, device="cuda")
>>>>>>> tmp

    # Create the KV cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device="cuda")
    key_cache.uniform_(-scale, scale)
<<<<<<< HEAD
    value_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape,
                              dtype=dtype,
                              device="cuda")
=======

    value_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size, block_size)
    value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device="cuda")
>>>>>>> tmp
    value_cache.uniform_(-scale, scale)

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
<<<<<<< HEAD
    if version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
=======
    
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
>>>>>>> tmp
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

<<<<<<< HEAD
    def run_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter_ns()

        for _ in range(num_iters):
            if version == "v1":
                ops.paged_attention_v1(
                    output,
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
            elif version == "v2":
                ops.paged_attention_v2(
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
                    block_size,
                    max_context_len,
                    alibi_slopes,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter_ns()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        elasped_time_ns = (end_time - start_time) / num_iters
        elasped_time_ms = elasped_time_ns / 1e6

        return elasped_time_ms

    # Warmup.
    print("Warming up...")
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency:.3f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(args)
=======
    def run_validation() -> bool:
        if version == "v1":
            ops.paged_attention_v1(
                validation_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables_validation,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
            )
            ops.paged_attention_v1_target(
                output,
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
        elif version == "v2":
            ops.paged_attention_v2(
                validation_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables_validation,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
            )
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
        else:
            raise ValueError(f"Invalid version: {version}")

        # Print output and validation 
        # print("Output : ", output)
        # print("Validation Output : ", validation_output)
        # (hj) Output format is [num_seqs x query_lens, num_query_heads, head_size]
        
        # Compare differences
        print("Max diff : ", torch.max(torch.abs(output - validation_output)))
        print("Mean diff : ", torch.mean(torch.abs(output - validation_output)))

        # check difference of output and validation_output is same as 0 
        validation_success = torch.allclose(output, validation_output)
        
        return validation_success

    def run_benchmark(target:bool, num_iters: int) -> float:
        
        # print("Running benchmark with shapes:")
        # print(f"  query: {query.shape}")
        # print(f"  key_cache: {key_cache.shape}")
        # print(f"  value_cache: {value_cache.shape}")
        
        # print("Query Information")
        # print("Num Seqs : ", num_seqs)
        # print("Query Lens : ", query_lens)
        # print("Context Lens : ", context_lens)

        def run():
            if version == "v1":
                if target:
                    ops.paged_attention_v1_target(
                        output,
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
                else:
                    ops.paged_attention_v1(
                        validation_output,
                        query,
                        key_cache,
                        value_cache,
                        head_mapping,
                        scale,
                        block_tables_validation,
                        context_lens,
                        block_size,
                        max_context_len,
                        alibi_slopes,
                    )
            elif version == "v2":
                if target:
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
                else:
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
                        block_tables_validation,
                        context_lens,
                        block_size,
                        max_context_len,
                        alibi_slopes,
                    )
            else:
                raise ValueError(f"Invalid version: {version}")
        
        # Warmup.
        for _ in range(3):
            run()

        torch.cuda.synchronize()
        start_time = time.perf_counter_ns()
        for _ in range(num_iters):
            run()
        torch.cuda.synchronize()
        end_time = time.perf_counter_ns()

        elasped_time_ns = (end_time - start_time) / num_iters
        elasped_time_ms = elasped_time_ns / 1e6
        
        return elasped_time_ms

    # Validation
    success = run_validation()
    print("Validation success: ", success)

    original_latency = run_benchmark(target=False, num_iters=100)
    target_latency = run_benchmark(target=True, num_iters=100)

    print(f"Original kernel running time: {original_latency:.3f} ms")
    print(f"Target kernel running time: {target_latency:.3f} ms")
    print(f"Speedup : {original_latency / target_latency:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel."
    )
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--batch-size", type=int, default=108)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument(
        "--head-size", type=int, choices=[64, 80, 96, 112, 128, 256], default=128
    )
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-len", type=int, default=-1) # -1 is random
    args = parser.parse_args()
>>>>>>> tmp

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }
<<<<<<< HEAD
    main(
        version=args.version,
        num_seqs=args.batch_size,
        context_len=args.context_len,
=======

    # For context len, do it for each query
    def gen_context_len(max_context_len_per_query : List[int], query_lens : List[int]):
        res = []
        for query_len, c_max in zip(query_lens, max_context_len_per_query):
            for i in range(query_len):
                res.append(c_max - (query_len - i - 1))
        return res
        
    if args.query_len == -1:
        query_lens = [random.randint(2, 16) for _ in range(args.batch_size)]
    else:
        query_lens = [args.query_len] * args.batch_size

    if args.context_len == -1:
        context_lens = gen_context_len([random.randint(256, 1024) for _ in range(args.batch_size)], query_lens)
    else:
        context_lens = gen_context_len([args.context_len] * args.batch_size, query_lens)    

    main(
        version=args.version,
        num_seqs=args.batch_size,
        context_lens=context_lens,
        query_lens=query_lens,
>>>>>>> tmp
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=dtype_to_torch_dtype[args.dtype],
<<<<<<< HEAD
        seed=args.seed,
        do_profile=args.profile,
=======
        seed=args.seed
>>>>>>> tmp
    )
