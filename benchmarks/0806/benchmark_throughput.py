"""Benchmark offline inference throughput."""

import argparse
import random
import time
import gc
from itertools import cycle
from typing import List, Optional, Tuple

import numpy as np
from tabulate import tabulate
from transformers import AutoTokenizer

from dataset import sample_requests
from vllm import LLM, SpecDecodeLLM, SamplingParams
from vllm.outputs import RequestOutput

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'

# Disable garbage collection for performance
gc.disable()

# Throughput (requests/sec, input_tokens/sec, output_tokens/sec):
# Measures the engine's capacity by tracking the number of requests, input tokens, and output tokens processed per second.
# TTFT (Time to First Token):
# Time to generate the first response token after receiving a request, including any queuing delays.
# TPOT (Time Per Output Token):
# Average time to generate each token after the first (excluding TTFT), also known as inter-token latency.
# Latency (End-to-End Processing Time):
# Total time to process a request, including both TTFT and TPOTs.
# Token Latency:
# Average time to generate each output token, calculated by dividing total latency by the number of output tokens.


def main(args: argparse.Namespace):
    random.seed(args.seed)

    # Display configuration tables
    config_table = [
        ["Target Model", args.target_model],
        ["Draft Model", args.draft_model],
        ["Draft Size", args.draft_size],
        ["Temperature", args.temperature],
        ["Colocate", args.colocate],
        ["Prefill Schedule Mode", args.prefill_schedule_mode],
        ["Budget Token", args.budget_token],
        ["Budget Seq", args.budget_seq],
        ["Drop Threshold", args.drop_threshold],
        ["Target Attention", args.target_attention],
        ["Batch Size", args.batch_size],
        ["Input Length", args.input_len],
        ["Output Length", args.output_len],
    ]
    print(tabulate(config_table))

    llm = SpecDecodeLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        colocate=args.colocate,
        prefill_schedule_mode=args.prefill_schedule_mode,
        target_attention=args.target_attention,
        drop_threshold=args.drop_threshold,
        max_num_batched_tokens=args.budget_token,
        max_num_seqs=args.budget_seq,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        download_dir=DOWNLOAD_DIR,
    )

    # Sample the requests
    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    for _ in range(1):
        start_time = time.perf_counter()
        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Latency: {latency:.3f} sec")

    llm.llm_engine.worker_executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request.")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request. Overrides the output length from the dataset.")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Temperature for sampling. -1 for random temperature.")
    parser.add_argument('--colocate', '-c', action='store_true')
    parser.add_argument('--prefill-schedule-mode', '-psm', choices=[
                        'prioritize_prefill', 'full_prefill', 'chunked_prefill', 'chunked_prefill_demote_draft'], default='full_prefill')
    parser.add_argument("--target-attention",
                        action="store_true", help="Use target attention.")
    parser.add_argument("--drop-threshold", '-dt', type=float,
                        default=0, help="Threshold for dropping token.")
    parser.add_argument('--budget-token', type=int, default=2048,
                        help='Maximum number of tokens for each batch.')
    parser.add_argument('--budget-seq', type=int, default=128,
                        help='Maximum number of sequences for each request.')
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization', '-q',
                        choices=['awq', 'gptq', 'squeezellm', None], default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int,
                        default=None, help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='Trust remote code from Hugging Face.')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Maximum length of a sequence (including prompt and output).')
    parser.add_argument('--dtype', type=str, default='auto', choices=[
                        'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'], help='Data type for model weights and activations.')
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager execution.")
    parser.add_argument("--request-rate", type=float,
                        default=4, help="Number of requests per second.")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    assert args.input_len is not None
    assert args.output_len is not None

    main(args)
