"""Benchmark offline inference throughput."""
import argparse
import random
import time
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Tuple
from transformers import (AutoTokenizer)
from dataset import sample_requests
from tabulate import tabulate
from heapq import heappush, heappop
import torch

from vllm import SpecDecodeLLM, SamplingParams
from vllm.outputs import RequestOutput

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

DOWNLOAD_DIR = '/mnt/sda/download'


def get_requests_with_time(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> List[Tuple[float, Tuple[str, int, int]]]:
    input_requests = iter(input_requests)
    requests_with_time = []
    current_time = 0.0

    for request in input_requests:
        requests_with_time.append((current_time, request))

        if request_rate == float("inf"):
            # If the request rate is infinity, then all requests are sent at time 0.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # Accumulate the interval to get the next request time.
        current_time += interval

    return requests_with_time


def run(
        llm: SpecDecodeLLM,
        num_requests: int,
        requests: List[Tuple[str, int, int]],
        request_rate: float):
    requests_with_time = get_requests_with_time(requests, request_rate)
    requests_with_time.sort()  # Ensure the list is sorted by time
    pbar = tqdm(total=num_requests)

    outputs: List[RequestOutput] = []
    start_time = time.perf_counter()
    request_times = {}  # Dictionary to store request start times

    request_index = 0
    while len(outputs) < num_requests:
        current_time = time.perf_counter() - start_time

        # Add requests to the engine if their scheduled time has passed
        while request_index < num_requests and requests_with_time[request_index][0] <= current_time:
            request_start_time, (prompt, prompt_len,
                                 output_len) = requests_with_time[request_index]
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=output_len,
            )
            request_id = llm._add_request(prompt=prompt,
                                          prompt_token_ids=None,
                                          sampling_params=sampling_params)
            request_times[request_id] = (
                request_start_time, prompt_len, output_len)
            request_index += 1

        step_outputs = llm.llm_engine.step()
        # print("batch size", len(step_outputs))
        for output in step_outputs:
            if output.finished:
                request_end_time = time.perf_counter() - start_time
                request_id = output.request_id
                request_start_time, prompt_len, output_len = request_times[request_id]
                request_latency = request_end_time - request_start_time
                REQUEST_LATENCY.append(
                    (prompt_len, output_len, request_latency))
                outputs.append(output)
                pbar.update(1)

    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    table = [["target_model", args.target_model],
             ["draft_model", args.draft_model],
             ["draft_size", args.draft_size],
             ["colocate", args.colocate],
             ["chunked_prefill", args.chunked_prefill],
             ["dataset", args.dataset],
             ["input_len", args.input_len],
             ["output_len", args.output_len],
             ["num_prompts", args.num_prompts],
             ["request_rate", args.request_rate],]

    print(tabulate(table))

    llm = SpecDecodeLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        colocate=args.colocate,
        enable_chunked_prefill=args.chunked_prefill,
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

    def warmup():
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=128,
        )
        dummy_prompt_token_ids = [[0] * 32] * 8
        start_time = time.perf_counter()
        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    warmup()

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                               args.output_len)
    benchmark_start_time = time.perf_counter()
    run(llm, args.num_prompts, requests, args.request_rate)
    benchmark_end_time = time.perf_counter()

    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.3f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.3f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.3f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.3f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.3f} s")

    llm.llm_engine.worker_executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-13b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--colocate',
                        '-c',
                        action='store_true')
    parser.add_argument('--chunked-prefill',
                        '-cp',
                        action='store_true')
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.target_model
    if args.dataset is None:
        args.dataset = "dummy"
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    main(args)
