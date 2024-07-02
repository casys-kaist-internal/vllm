"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
from pathlib import Path
from typing import Optional
from tabulate import tabulate
from transformers import (AutoTokenizer)
from dataset import sample_requests

import numpy as np
import torch
from tqdm import tqdm

from vllm import SpecDecodeLLM, SamplingParams

DOWNLOAD_DIR = '/home/noppanat/workspace/models'


def main(args: argparse.Namespace):
    print(args)

    table = [["target_model", args.target_model],
             ["draft_model", args.draft_model],
             ["draft_size", args.draft_size],
             ["colocate", args.colocate],
             ["chunked_prefill", args.chunked_prefill],
             ["dataset", args.dataset],
             ["input_len", args.input_len],
             ["output_len", args.output_len],
             ["batch_size", args.batch_size],]

    print(tabulate(table))

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = SpecDecodeLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        colocate=args.colocate,
        enable_chunked_prefill=args.chunked_prefill,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        download_dir=DOWNLOAD_DIR,
        # disable_log_stats=False
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)

    if args.dataset == "dummy":
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.batch_size)]
    else:
        requests = sample_requests(args.dataset, args.batch_size, tokenizer,
                                   args.output_len)

    def run_to_completion():
        # Add the requests to the engine.
        for prompt, _, output_len in requests:
            sampling_params = SamplingParams(
                n=args.n,
                temperature=0.0,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=output_len,
            )
            # FIXME(woosuk): Do not use internal method.
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
            )

        start = time.perf_counter()
        # FIXME(woosuk): Do not use internal method.
        llm._run_engine(use_tqdm=False)
        end = time.perf_counter()

        return end - start

    print("Warming up...")
    run_to_completion()

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion())
    print(f'Avg latency: {np.mean(latencies)} seconds')

    llm.llm_engine.worker_executor.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--colocate',
                        '-c',
                        action='store_true')
    parser.add_argument('--chunked-prefill',
                        '-cp',
                        action='store_true')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    if args.dataset is None:
        args.dataset = "dummy"
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        args.input_len = "N/A"

    main(args)
