import argparse
import random
import time
import math
import torch
from vllm._C import ops
from typing import List
import socket

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
from vllm.model_executor.input_metadata import InputMetadata


NUM_BLOCKS = 8192
PARTITION_SIZE = 512

download_dir = "/workspace/vllm/.huggingface_cache"
import os

os.environ["HF_HOME"] = "/workspace/vllm/.huggingface_cache"


def gen_context_len(max_context_len_per_query: List[int], query_lens: List[int]):
    res = []
    for query_len, c_max in zip(query_lens, max_context_len_per_query):
        for i in range(query_len):
            res.append(c_max - (query_len - i - 1))
    return res


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device="cuda")


@torch.inference_mode()
def main(args) -> None:
    from vllm import LLM, SpSLLM, SamplingParams

    global download_dir

    target_model = args.target_model
    draft_model = args.draft_model
    draft_size = args.draft_size
    tokenizer = args.tokenizer
    quantization = args.quantization
    tensor_parallel_size = args.tensor_parallel_size
    seed = args.seed
    trust_remote_code = args.trust_remote_code
    dtype = args.dtype

    llm = SpSLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        tile_size=args.tile_size,
        use_dynamic_draft_size=args.dynamic_draft,
        use_tile_size_constraint=args.use_tile_size,
        use_lazy_draft_kv_cache=True,
        use_target_attention=args.use_target_attention,
        target_draft_latency_ratio=args.target_draft_latency_ratio,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        download_dir=download_dir,
    )

    # print("hei")
    llm_engine = llm.llm_engine
    worker = llm_engine.workers[0]
    target_model_runner = worker.target_model_runner
    target_model = target_model_runner.model

    decoder = target_model.model.decoder
    layer = decoder.layers[0]
    # hidden_states : [batch_size x max_query_len x dmodel]

    time_results = []
    CONTEXT_LEN = 256

    # for batch_size in range(32, 512, 4):
    for batch_size in [32]:

        max_query_len = 8
        dmodel = 4096  # todo: get from model

        target_gpu_cache = worker.target_gpu_cache
        kv_cache = target_gpu_cache

        print(kv_cache[0][1].shape)

        # generate random input metadata
        prompt_lens = []
        draft_lens = []

        block_size = worker.cache_config.block_size

        num_blocks_per_layer = (CONTEXT_LEN + 10) // block_size + 1

        def gen_tok_lens(avg, batch_size):
            # generate by normal distribution, min 1, max 8
            target_lens = []
            for i in range(batch_size):
                sam = math.ceil(random.normalvariate(avg, 3))
                sam = max(1, sam)
                sam = min(8, sam)
                target_lens.append(sam)
            return target_lens

        # for input_tok_len in range(batch_size, 225):
        # for input_tok_len in range(batch_size, batch_size * 8 + 1, 1):
        for avg_input_tok_len in range(batch_size, batch_size * 8 + 1, 1):
            print(f"Input token length: {avg_input_tok_len}")

            input_tokens = []
            input_positions = []

            target_lens = gen_tok_lens(avg_input_tok_len / batch_size, batch_size)
            input_tok_len = sum(target_lens)

            # Add arbitrary draft_size.
            # avg_draft_len = int(input_tok_len / batch_size)
            # target_lens = [avg_draft_len for _ in range(batch_size)]
            # num_toks_to_add = input_tok_len - sum(target_lens)
            # for i in range(num_toks_to_add):
            #     target_lens[i] += 1


            context_lens_perbatch = [
                random.randint(32, CONTEXT_LEN) + target_lens[i]
                for i in range(batch_size)
            ]  # TODO: parameterise]

            print(context_lens_perbatch)
            context_lens = gen_context_len(context_lens_perbatch, target_lens)
            block_tables = []
            bt_idx = 0
            slot_mapping = []
            for seq_idx in range(batch_size):  # (hj) : Valid in target context too!
                # Sequence 사이에는 공유하는 K/V 없다고 가정
                block_table = [i for i in range(bt_idx, bt_idx + num_blocks_per_layer)]
                bt_idx += num_blocks_per_layer

                position_start = context_lens_perbatch[seq_idx]

                for position in range(
                    position_start, position_start + max(target_lens)
                ):
                    if position < position_start + target_lens[seq_idx]:
                        input_tokens.append([random.randint(0, 10000)])
                        input_positions.append([position])

                        block_number = block_table[position // block_size]
                        block_offset = position % block_size
                        slot = block_number * block_size + block_offset
                        slot_mapping.append([slot])
                        block_tables.append(block_table)
                    else:
                        slot_mapping.append([-1])

                # for remaining, (until block edge) pad slot with -1

            print(target_lens)

            slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cuda")
            max_context_len = max(context_lens)
            context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
            query_lens = torch.tensor(target_lens, dtype=torch.int, device="cuda")
            max_block_table_len = max([len(t) for t in block_tables])
            block_tables = _make_tensor_with_pad(
                block_tables, max_len=max_block_table_len, pad=0, dtype=torch.int
            )

            print(slot_mapping.shape)

            # assert slot_mapping.shape[0] == input_tok_len

            input_metadata = InputMetadata(
                prompt_lens=[],
                draft_lens=[],
                target_lens=target_lens,
                slot_mapping=slot_mapping,
                max_context_len=max_context_len,
                context_lens=context_lens,
                query_lens=query_lens,
                block_tables=block_tables,
                use_target_attention=True,
            )
            REPEAT = 1
            input_ids = torch.tensor(input_tokens, dtype=torch.long, device="cuda")
            input_positions = torch.tensor(
                input_positions, dtype=torch.long, device="cuda"
            )

            # sample_input = torch.rand((input_tok_len, ,dmodel ), device="cuda")

            # Run the layer
            start = time.perf_counter()
            for _ in range(REPEAT):
                res = decoder.forward(
                    input_ids, input_positions, kv_cache, input_metadata, None
                )
                # res = layer.forward(
                #     sample_input,
                #     input_metadata,
                #     kv_cache,
                #     None,
                # )
            torch.cuda.synchronize()
            end = time.perf_counter()

            time_taken = (end - start) / REPEAT

            results = {
                "input_tok_len": input_tok_len,
                "time_taken": time_taken,
                "max_query_len": max(target_lens),
                "throughput": input_tok_len / time_taken,
                "target_lens": target_lens,
            }
            time_results.append(results)

    # Save time_results as json
    import json

    with open(f"layer_tiling_results_decoderlayer_{args.gpunum}.json", "w") as f:
        json.dump(time_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--engine", type=str, choices=["base", "sps"], default="base")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "humaneval", "alpaca", "mt-bench", "sharegpt", "apps"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        # default='EleutherAI/pythia-12b')
        # default="facebook/opt-6.7b",
        default="facebook/opt-2.7b",
    )
    # default = 'facebook/opt-2.7b')
    # default='bigscience/bloom-7b1')
    # default='daryl149/llama-2-7b-chat-hf')
    # default='facebook/opt-6.7b')
    parser.add_argument(
        "--draft-model",
        type=str,
        # default='EleutherAI/pythia-410m')
        # default='bigscience/bloomz-560m')
        # default='Felladrin/Llama-68M-Chat-v1')
        default="facebook/opt-125m",
    )
    parser.add_argument("--draft-size", type=int, default=7)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--dynamic-draft", action="store_true")
    parser.add_argument("--use-tile-size", action="store_true")
    parser.add_argument("--use-lazy-draft-kv-cache", action="store_true")
    parser.add_argument("--use-target-attention", action="store_true")
    parser.add_argument("--target-draft-latency-ratio", "-c", type=float, default=0.2)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--quantization", "-q", choices=["awq", "squeezellm", None], default=None
    )
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.5, help="Sampling temperature."
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to process."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-iters", type=int, default=1, help="Number of iterations to run."
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="data type for model weights and activations. "
        'The "auto" option will use FP16 precision '
        "for FP32 and FP16 models, and BF16 precision "
        "for BF16 models.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--gpunum",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.target_model
    main(args)
