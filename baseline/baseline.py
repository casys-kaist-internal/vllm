"""Benchmark offline inference throughput."""

import argparse
import random
import time
import gc
from itertools import cycle
from typing import List, Optional, Tuple
import json

import numpy as np
from tabulate import tabulate
from transformers import AutoTokenizer
import torch
from dataset import sample_requests 

from vllm import LLM,  SamplingParams
from vllm.outputs import RequestOutput
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'
BENCHMARK_DURATION_IN_MINUTES = 5

# Disable garbage collection for performance
gc.disable()

if torch.cuda.is_available():
    gpu_index = 0  # First GPU
    gpu_name = torch.cuda.get_device_name(gpu_index)
    print(gpu_name)
else:
    print("No CUDA device available")

def get_requests_with_time(input_requests: List[Tuple[str, int, int]],
                           request_rate: float) -> List[Tuple[float, Tuple[str, int, int]]]:
    """Generates requests with associated times based on a Poisson process."""
    requests_with_time = []
    current_time = 0.0

    for request in cycle(input_requests):
        requests_with_time.append((current_time, request))
        interval = np.random.exponential(1.0 / request_rate)
        current_time += interval

        # Add 1 minute to the benchmark duration for safety
        if current_time > (BENCHMARK_DURATION_IN_MINUTES + 1) * 60:
            break

    return requests_with_time


def run(llm: LLM, requests: List[Tuple[str, int, int]], request_rate: float, temperature: float) -> Tuple[dict, int, bool]:
    """Runs the benchmark, processing requests with the given LLM."""
    requests_with_time = get_requests_with_time(requests, request_rate)
    outputs: List[RequestOutput] = []
    result = {}

    start_time = time.perf_counter()

    request_index = 0
    while time.perf_counter() - start_time < BENCHMARK_DURATION_IN_MINUTES * 60:
        current_time = time.perf_counter() - start_time

        # Add requests to the engine if their scheduled time has passed
        while requests_with_time[request_index][0] <= current_time:
            request_start_time, (prompt, prompt_len,
                                 output_len) = requests_with_time[request_index]
            sampling_params = SamplingParams(
                n=1,
                temperature=random.choice(
                    [0, 0.25, 0.5, 0.75]) if temperature == -1 else temperature,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=output_len,
            )
            request_id = llm._add_request(
                inputs=prompt, params=sampling_params)
            result[str(request_id)] = [request_start_time]
            request_index += 1

        step_outputs = llm.llm_engine.step()
        for output in step_outputs:
            if len(output.outputs[0].token_ids) == 1 and len(result[output.request_id]) == 1:
                ttft = time.perf_counter() - start_time - \
                    result[output.request_id][0]
                result[output.request_id].append(ttft)

            if output.finished:
                e2e_latency = time.perf_counter() - start_time - \
                    result[output.request_id][0]
                result[output.request_id].extend(
                    [e2e_latency, len(output.prompt_token_ids), len(output.outputs[0].token_ids)])
                outputs.append(output)

        throughput = len(outputs) / (time.perf_counter() - start_time)
    print(f"Throughput: {throughput:.3f} reqs/s")

    # remove request_id from result if not exist in outputs
    for request_id in list(result.keys()):
        if request_id not in [output.request_id for output in outputs]:
            del result[request_id]

    total_tokens = sum(prompt_len + output_len for _, _, _,
                       prompt_len, output_len in result.values())

    return result, total_tokens

def analyze_results(result: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyzes the results to compute TTFT, TPOT, and token latencies."""
    ttfts, tpots, token_latencies = [], [], []

    for _, values in result.items():
        _, ttft, e2e_latency, _, output_len = values
        ttfts.append(ttft)
        tpots.append((e2e_latency - ttft) / (output_len - 1))
        token_latencies.append(e2e_latency / output_len)

    return np.array(ttfts), np.array(tpots), np.array(token_latencies)


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
        ["Dataset", args.dataset],
        ["Request Rate", args.request_rate],
    ]
    print(tabulate(config_table))
    llm = LLM(
        model=args.target_model,
        speculative_model=args.draft_model,
        num_speculative_tokens=args.draft_size,
        use_v2_block_manager=True,
        gpu_memory_utilization=0.85,
    )

    # Sample the requests
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    # 0 indicates all requests
    requests = sample_requests(args.dataset, 0, tokenizer)

    # Run the benchmark
    start_time = time.perf_counter()
    result, total_tokens = run(
        llm, requests, args.request_rate, args.temperature)
    elapsed_time = time.perf_counter() - start_time

    # Analyze results
    ttfts, tpots, token_latencies = analyze_results(result)

    # Main results
    request_throughput = len(result) / elapsed_time
    token_throughput = total_tokens / elapsed_time
    # token_latency: the average processing time per output token
    token_latency = np.mean(token_latencies)

    # Sub results
    p50_ttft = np.percentile(ttfts, 50)
    p99_ttft = np.percentile(ttfts, 99)
    p50_tpot = np.percentile(tpots, 50)
    p99_tpot = np.percentile(tpots, 99)
    p50_token_latency = np.percentile(token_latencies, 50)
    p99_token_latency = np.percentile(token_latencies, 99)

    #remove spaces in gpu_name
    gpu_index = 0  # First GPU
    gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_name = gpu_name.replace(" ", "")

    # print("GPU Name,Target Model, Draft Model, Dataset,Temperature,Request Rate,Draft Size,Request Throughput (reqs/s),Token Throughput (tokens/s),Token Latency (s/token),P50 TTFT (s),P99 TTFT (s),P50 TPOT (s/token),P99 TPOT (s/token),P50 Token Latency (s/token),P99 Token Latency (s/token), Disable by Batch Size")
    # print(f"Result,{gpu_name},{args.target_model},{args.draft_model},{args.dataset},{args.temperature},{args.request_rate},{args.draft_size},{request_throughput},{token_throughput},{token_latency},{p50_ttft},{p99_ttft},{p50_tpot},{p99_tpot},{p50_token_latency},{p99_token_latency},False")
    print(f"Result, {request_throughput:.3f}, {token_throughput:.3f}, {token_latency:.6f}, {p50_ttft:.6f}, {p99_ttft:.6f}, {p50_tpot:.6f}, {p99_tpot:.6f}, {p50_token_latency:.6f}, {p99_token_latency:.6f}, False")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default='sharegpt',
                        help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request.")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request. Overrides the output length from the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=4)
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
    parser.add_argument('--budget-seq', type=int, default=64,
                        help='Maximum number of sequences for each request.')
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization', '-q',
                        choices=['awq', 'gptq', 'squeezellm', None], default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
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
    parser.add_argument("--speculative-disable-by-batch-size", type=int, default=4)

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    if args.dataset is None or args.dataset == "dummy":
        args.dataset = "dummy"

    main(args)