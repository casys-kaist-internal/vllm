# Benchmark offline inference throughput with SLO considerations.

import argparse
import random
import time
import gc
from itertools import cycle
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from transformers import AutoTokenizer

from dataset import sample_requests
from vllm import SpecDecodeLLM, SamplingParams
from vllm.outputs import RequestOutput

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'
BENCHMARK_DURATION_IN_MINUTES = 1

# Disable garbage collection for performance
gc.disable()

# Global variable to decide the arrival pattern
ARRIVAL_PATTERN = 'poisson'  # Options: 'poisson' or 'bursty'

TPOT_STRICT_SLO = 0.1  # Hardcoded TPOT SLO for strict scenario (seconds)
TPOT_RELAXED_SLO = 0.5  # Hardcoded TPOT SLO for relaxed scenario (seconds)

def get_requests_with_time(input_requests: List[Tuple[str, int, int]],
                           request_rate: float) -> List[Tuple[float, Tuple[str, int, int]]]:
    """
    Generates requests with associated times based on the selected arrival pattern.
    The arrival pattern is controlled by the global variable ARRIVAL_PATTERN.
    """
    requests_with_time = []
    current_time = 0.0
    total_duration = (BENCHMARK_DURATION_IN_MINUTES + 1) * 60  # Total benchmark duration in seconds

    if ARRIVAL_PATTERN == 'poisson':
        # Poisson arrival process
        for request in cycle(input_requests):
            requests_with_time.append((current_time, request))
            interval = np.random.exponential(1.0 / request_rate)
            current_time += interval

            # Stop if we've reached the total duration
            if current_time > total_duration:
                break

    elif ARRIVAL_PATTERN == 'bursty':
        # Bursty arrival pattern
        # Parameters for burstiness
        burst_interval = 10  # Seconds between the start of each burst
        burst_duration = 1  # Duration of each burst in seconds
        high_rate_multiplier = 5  # Multiplier for the arrival rate during bursts

        # Create an infinite cycle of input requests
        input_requests_cycle = cycle(input_requests)

        # Time when the next burst starts
        next_burst_start = 0.0

        while current_time < total_duration:
            if next_burst_start <= current_time < next_burst_start + burst_duration:
                # We are in a burst period
                current_request_rate = request_rate * high_rate_multiplier
            else:
                # We are in a normal period
                current_request_rate = request_rate

                # If we've passed the burst period, schedule the next burst
                if current_time >= next_burst_start + burst_duration:
                    next_burst_start += burst_interval

            # Generate the inter-arrival interval based on the current request rate
            if current_request_rate > 0:
                interval = np.random.exponential(1.0 / current_request_rate)
            else:
                # Avoid division by zero if the rate is zero
                interval = float('inf')

            requests_with_time.append((current_time, next(input_requests_cycle)))
            current_time += interval

    else:
        raise ValueError(f"Invalid ARRIVAL_PATTERN: {ARRIVAL_PATTERN}. Must be 'poisson' or 'bursty'.")

    return requests_with_time


def run(llm: SpecDecodeLLM, requests: List[Tuple[str, int, int]], request_rate: float, temperature: float) -> Tuple[dict, int, bool]:
    """Runs the benchmark, processing requests with the given LLM."""
    requests_with_time = get_requests_with_time(requests, request_rate)
    outputs: List[RequestOutput] = []
    result = {}

    start_time = time.perf_counter()
    llm.llm_engine.reset_total_tokens()
    start_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()

    total_processed_tokens = 0
    request_index = 0
    total_requests = len(requests_with_time)

    while time.perf_counter() - start_time < BENCHMARK_DURATION_IN_MINUTES * 60:
        current_time = time.perf_counter() - start_time

        # Add requests to the engine if their scheduled time has passed
        while request_index < total_requests and requests_with_time[request_index][0] <= current_time:
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
                prompt=prompt, prompt_token_ids=None, sampling_params=sampling_params)
            result[request_id] = [request_start_time]
            request_index += 1

            # If all requests have been scheduled, break
            if request_index >= total_requests:
                break

        step_outputs = llm.llm_engine.step()

        for output in step_outputs:
            if len(output.outputs[0].token_ids) == 1 and len(result[output.request_id]) == 1:
                ttft = time.perf_counter() - start_time - result[output.request_id][0]
                result[output.request_id].append(ttft)

            if output.finished:
                e2e_latency = time.perf_counter() - start_time - result[output.request_id][0]
                result[output.request_id].extend(
                    [e2e_latency, len(output.prompt_token_ids), len(output.outputs[0].token_ids)])
                outputs.append(output)
                total_processed_tokens += (len(output.prompt_token_ids) +
                                           len(output.outputs[0].token_ids))

    # Remove incomplete requests from result
    for request_id in list(result.keys()):
        if request_id not in [output.request_id for output in outputs]:
            del result[request_id]

    total_tokens = sum(prompt_len + output_len for _, _, _, prompt_len, output_len in result.values())

    llm.llm_engine.abort_all_requests()
    llm.llm_engine.worker_executor.shutdown()

    end_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()
    assert start_num_free_gpu_blocks == end_num_free_gpu_blocks

    return result, total_tokens, llm.llm_engine.scheduler.preempt_flag


def analyze_results(result: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Analyzes the results to compute TTFT and TPOT."""
    ttfts, tpots = [], []

    for _, values in result.items():
        _, ttft, e2e_latency, _, output_len = values
        ttfts.append(ttft)
        if output_len > 1:
            tpots.append((e2e_latency - ttft) / (output_len - 1))
        else:
            tpots.append(0)

    return np.array(ttfts), np.array(tpots)


def find_max_capacity(llm, requests, temperature, tpot_slo, slo_name):
    """Finds the maximum request rate that meets the given TPOT SLO while monitoring TTFT."""
    request_rate = 10  # Start with 1 req/sec
    max_request_rate = None
    increment = 1  # Adjust as needed
    max_rate_reached = False
    previous_ttft = None
    sharp_increase_threshold = 2.0  # Factor by which TTFT increase is considered sharp

    while not max_rate_reached:
        print(f"\nTesting {slo_name} SLO at {request_rate} req/sec")

        # Run the benchmark
        result, _, _ = run(llm, requests, request_rate, temperature)
        ttfts, tpots = analyze_results(result)
        p99_ttft = np.percentile(ttfts, 99)
        p99_tpot = np.percentile(tpots, 99)

        print(f"P99 TTFT: {p99_ttft:.3f} s, P99 TPOT: {p99_tpot:.3f} s (TPOT SLO: {tpot_slo:.3f} s)")

        if p99_tpot <= tpot_slo:
            if previous_ttft is not None and p99_ttft > sharp_increase_threshold * previous_ttft:
                # Sharp increase in TTFT detected, stop and consider previous rate as max capacity
                max_rate_reached = True
            else:
                max_request_rate = request_rate
                previous_ttft = p99_ttft
                request_rate += increment
        else:
            max_rate_reached = True

    return max_request_rate


def plot_capacity(results):
    # Results is a list of dictionaries with keys: 'slo', 'max_capacity'
    slo_names = [res['slo'] for res in results]
    capacities = [res['max_capacity'] for res in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(slo_names, capacities, color=['#1f77b4', '#ff7f0e'])

    for bar, capacity in zip(bars, capacities):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom')

    plt.xlabel('SLO Type')
    plt.ylabel('Maximum Sustainable Request Rate (req/sec)')
    plt.title('Maximum Capacity under Different SLOs')
    plt.tight_layout()
    plt.savefig('capacity_vs_slo.png')
    plt.show()

def warmup(llm):
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
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

def train_predictor(llm: SpecDecodeLLM, requests: List[Tuple[str, int, int]], request_rate: float, temperature: float):
    """Train the predictor."""
    print("Training the predictor for selective validation ...")

    # Change the scheduler_config to full_prefill for training predictor
    original_chunked_prefill = llm.llm_engine.scheduler_config.chunked_prefill
    orignal_max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    llm.llm_engine.scheduler_config.chunked_prefill = False
    llm.llm_engine.scheduler_config.max_num_batched_tokens = 2048

    requests_with_time = get_requests_with_time(requests, request_rate)
    result = {}
    total_requests = len(requests_with_time)

    start_time = time.perf_counter()
    llm.llm_engine.reset_total_tokens()

    request_index = 0
    overall_start_time = start_time  # Keep track of the overall start time
    while True:
        current_time = time.perf_counter() - start_time

        # Add requests to the engine if their scheduled time has passed
        while request_index < total_requests and requests_with_time[request_index][0] <= current_time:
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
                prompt=prompt, prompt_token_ids=None, sampling_params=sampling_params)
            result[request_id] = [request_start_time]
            request_index += 1

        # Reset request_index if all requests have been processed to continue
        if request_index >= total_requests:
            request_index = 0
            start_time = time.perf_counter()  # Update start_time for the next cycle

        llm.llm_engine.step()

        if llm.llm_engine.scheduler.accept_prob_predictor.is_trained():
            break

    end_time = time.perf_counter()
    elapsed_time = end_time - overall_start_time  # Calculate elapsed time from the overall start time

    print(f"Predictor training time: {elapsed_time:.3f} s")

    llm.llm_engine.scheduler_config.chunked_prefill = original_chunked_prefill
    llm.llm_engine.scheduler_config.max_num_batched_tokens = orignal_max_num_batched_tokens
    llm.llm_engine.abort_all_requests()


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
        ["Selective Validation", args.selective_validation],
        ["Drop Threshold", args.drop_threshold],
        ["Consolidated Attention", args.consolidated_attention],
        ["Dataset", args.dataset],
    ]
    print(tabulate(config_table))

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    # Initialize LLM
    llm = SpecDecodeLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        colocate=args.colocate,
        prefill_schedule_mode=args.prefill_schedule_mode,
        consolidated_attention=args.consolidated_attention,
        selective_validation=args.selective_validation,
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
        gpu_memory_utilization=0.85
    )

    if args.selective_validation:
        requests = sample_requests(args.dataset, 0, tokenizer)
        # For training predictor, set request_rate high enough to train quickly
        train_predictor(llm, requests, 16, args.temperature)
    else:
        warmup(llm)

    # Load all requests
    requests = sample_requests(args.dataset, 0, tokenizer)

    # Find maximum capacity under strict SLO
    max_capacity_strict = find_max_capacity(
        llm, requests, args.temperature, TPOT_STRICT_SLO, "Strict")

    # Find maximum capacity under relaxed SLO
    max_capacity_relaxed = find_max_capacity(
        llm, requests, args.temperature, TPOT_RELAXED_SLO, "Relaxed")

    # Collect results for plotting
    all_results = [
        {'slo': 'Strict', 'max_capacity': max_capacity_strict},
        {'slo': 'Relaxed', 'max_capacity': max_capacity_relaxed}
    ]

    # Plot the capacity
    plot_capacity(all_results)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the throughput with SLO considerations.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='EleutherAI/pythia-6.9b')
    parser.add_argument('--draft-model', type=str, default='EleutherAI/pythia-160m')
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Temperature for sampling. -1 for random temperature in [0, 0.25, 0.5, 0.75].")
    parser.add_argument('--colocate', '-c', action='store_true')
    parser.add_argument('--prefill-schedule-mode', '-psm', choices=[
                        'full_prefill', 'chunked_prefill'], default='full_prefill')
    parser.add_argument("--consolidated-attention",
                        action="store_true", help="Use consolidated attention.")
    parser.add_argument("--selective-validation", action="store_true")
    parser.add_argument("--drop-threshold", '-dt', type=float,
                        default=0, help="Threshold for dropping token.")
    parser.add_argument('--budget-token', type=int, default=512,
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
    # Removed --request-rates since we're finding max capacity
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
