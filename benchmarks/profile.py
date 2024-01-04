"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import time
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)
from vllm import LLM, SpSLLM, SamplingParams

download_dir = '/home/sjchoi/workspace/models'
MAX_KV_CACHE = 10000


def main(args: argparse.Namespace):
    global download_dir
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.

    llm = LLM(
        model=args.target_model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        download_dir=download_dir
    )

    dummy_prompt_token_ids = [
        random.randint(0, 50272) for _ in range(1)]

    for batch_size in range(1, 257):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=2048
        )

        for _ in range(batch_size):
            llm._add_request(
                prompt=None,
                prompt_token_ids=dummy_prompt_token_ids,
                sampling_params=sampling_params,
            )

        llm._run_engine(use_tqdm=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument("--engine", type=str, choices=["base", "sps"],
                        default="base")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature',
                        '-t',
                        type=float,
                        default=0,
                        help='Sampling temperature.')
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--num-iters',
                        type=int,
                        default=1,
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
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.target_model
    main(args)
