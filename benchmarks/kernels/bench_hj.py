import argparse
import random
import time
import math

import torch

from vllm._C import ops

from typing import List

NUM_BLOCKS = 81920
PARTITION_SIZE = 512

# too_many_blocks exception define


class TooManyBlocks(Exception):
    pass


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
    seed: int
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))

    # (hj) Query format is [num_seqs x query_lens, num_query_heads, head_size]

    sum_query_lens = sum(query_lens)  # Query is jagged tensor of above shape

    query = torch.empty(
        sum_query_lens, num_query_heads, head_size, dtype=dtype, device="cuda"
    )

    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32,
                     device="cuda"), num_queries_per_kv
    )
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(
            num_query_heads, dtype=torch.float, device="cuda")

    # (hj) Changed as we are passing context_lens as input
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
    max_context_len = context_lens.max().item()

    query_lens = torch.tensor(query_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []  # For original kernel
    total_num_blocks_needed = max_num_blocks_per_seq * num_seqs
    if total_num_blocks_needed > NUM_BLOCKS:
        raise TooManyBlocks(
            f"Too many blocks needed: {total_num_blocks_needed} > {NUM_BLOCKS}")
    bt_idx = 0
    for seq_idx in range(num_seqs):  # (hj) : Valid in target context too!
        # Sequence 사이에는 공유하는 K/V 없다고 가정
        block_table = [i for i in range(
            bt_idx, bt_idx + max_num_blocks_per_seq)]
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
    value_cache = torch.empty(size=value_cache_shape,
                              dtype=dtype, device="cuda")
    value_cache.uniform_(-scale, scale)

    # for each value_cache block
    # for i in range(NUM_BLOCKS):
    # for j in range(num_kv_heads):
    for k in range(head_size):
        value_cache[:, :, k, :].fill_(k + 1)

    # print(value_cache[2, 0, :, :])

    ##########
    ##########
    ##########
    ##########
    ##########

    # Fill with 1
    query = torch.ones_like(query)
    key_cache = torch.ones_like(key_cache)
    for k in range(head_size):
        value_cache[:, :, k, :].fill_(k + 1)
    value_cache.uniform_(-scale, scale)

    ##########
    ##########
    ##########
    ##########
    ##########

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)

    # (hj) Another output to validate our stuff
    validation_output = torch.empty_like(query)

    if version == "v2":
        num_partitions = (max_context_len +
                          PARTITION_SIZE - 1) // PARTITION_SIZE

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

    def run_validation() -> bool:
        if version == "v1":
            ops.paged_attention_v1(
                validation_output,
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
        else:
            raise ValueError(f"Invalid version: {version}")

        # Print output and validation
        # print("Output : ", output)
        # print("Validation Output : ", validation_output)
        # (hj) Output format is [num_seqs x query_lens, num_query_heads, head_size]

        # Compare differences
        # print("Max diff : ", torch.max(torch.abs(output - validation_output)))
        # print("Mean diff : ", torch.mean(torch.abs(output - validation_output)))

        for head in range(0, 1):
            for s in range(num_seqs):
                for q in range(query_lens[s]):
                    print("Seq : ", s, " Head : ", head, " Query : ", q)
                    out = output[s * query_lens[s] + q, head]
                    val = validation_output[s * query_lens[s] + q, head]
                    # to list
                    out = out.tolist()
                    val = val.tolist()
                    # Print both, format to 3 decimal places
                    r1 = [round(x, 7) for x in out]
                    r2 = [round(x, 7) for x in val]

                    # Print both, format to 3 decimal places
                    # All values formatted to fit in 10 space
                    for i in range(len(r1)):
                        print(f"{r1[i]:<10} {r2[i]:<10}")

        print("-------------------")
        print("-------------------")
        print("-------------------")
        print("-------------------")
        print("-------------------")

        for head in range(0, 1):
            for s in range(num_seqs):
                for q in range(query_lens[s]):
                    for part in range(num_partitions):
                        print("Seq : ", s, " Head : ", head, " Query : ", q)
                        out = tmp_output_target[s *
                                                query_lens[s] + q, head, part]
                        val = tmp_output[s * query_lens[s] + q, head, part]
                        # to list
                        out = out.tolist()
                        val = val.tolist()
                        # Print both, format to 3 decimal places
                        r1 = [round(x, 7) for x in out]
                        r2 = [round(x, 7) for x in val]

                        # Print both, format to 3 decimal places
                        # All values formatted to fit in 10 space
                        print("Part : ", part)
                        for i in range(len(r1)):
                            print(f"{r1[i]:<10} {r2[i]:<10}")

        if not torch.allclose(output, validation_output):
            raise ValueError("Validation failed")

        # check difference of output and validation_output is same as 0
        validation_success = torch.allclose(output, validation_output)

        return validation_success

    def run_benchmark(target: bool, num_iters: int) -> float:

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
                        block_tables,
                        context_lens,
                        block_size,
                        max_context_len,
                        alibi_slopes,
                    )
            elif version == "v2":
                if target:
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
                        block_tables,
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
    torch.cuda.synchronize()
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
    parser.add_argument("--version", type=str,
                        choices=["v1", "v2"], default="v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=2048)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-len", type=int, default=8)  # -1 is random
    args = parser.parse_args()

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }

    # For context len, do it for each query
    def gen_context_len(max_context_len_per_query: List[int], query_lens: List[int]):
        res = []
        for query_len, c_max in zip(query_lens, max_context_len_per_query):
            for i in range(query_len):
                res.append(c_max - (query_len - i - 1))
        return res

    if args.query_len == -1:
        query_lens = [random.randint(2, 8) for _ in range(args.batch_size)]
    else:
        query_lens = [args.query_len] * args.batch_size

    if args.context_len == -1:
        context_lens = gen_context_len(
            [random.randint(256, 1024) for _ in range(args.batch_size)], query_lens)
    else:
        context_lens = gen_context_len(
            [args.context_len] * args.batch_size, query_lens)

    # Failed with query_len :  2  context_len :  65  num_seqs :  1

    print("CURRENT CONFIG")
    print("Version : ", args.version)
    print("Batch Size : ", args.batch_size)
    print("Context Lens : ", context_lens)
    print("Query Lens : ", query_lens)

    main(
        version=args.version,
        num_seqs=args.batch_size,
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

    # FUZZING
    # for query_len in range(1,16):
    # for context_len in range(query_len,1024,15):
    # for num_seqs in range(1, 150, 1): # Realistically will not go up to 150
    #     # query_lens = [query_len] * num_seqs
    #     query_lens = [random.randint(1, 16) for _ in range(num_seqs)]
    #     # context_lens = gen_context_len([context_len] * num_seqs, query_lens)
    #     context_lens = gen_context_len([random.randint(256, 4096) for _ in range(num_seqs)], query_lens)
    #     try :
    #         main(
    #             version=args.version,
    #             num_seqs=num_seqs,
    #             context_lens=context_lens,
    #             query_lens=query_lens,
    #             num_query_heads=args.num_query_heads,
    #             num_kv_heads=args.num_kv_heads,
    #             head_size=args.head_size,
    #             block_size=args.block_size,
    #             use_alibi=args.use_alibi,
    #             dtype=dtype_to_torch_dtype[args.dtype],
    #             seed=args.seed
    #         )
    #         average_query_len = sum(query_lens) / num_seqs
    #         average_context_len = sum(context_lens) / num_seqs
    #         print("PASS query_len : ", average_query_len, " context_len : ", average_context_len, " num_seqs : ", num_seqs)
    #     except TooManyBlocks as e:
    #         print("SKIP : ", average_query_len, " context_len : ", average_context_len, " num_seqs : ", num_seqs)
    #     except Exception as e:
    #         print("Failed with query_len : ", average_query_len, " context_len : ", average_context_len, " num_seqs : ", num_seqs)
    #         print(e)
    #         break
