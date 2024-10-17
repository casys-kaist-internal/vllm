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

from vllm import LLM,  SamplingParams
from vllm.outputs import RequestOutput
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'

# Disable garbage collection for performance
gc.disable()

# Global variables for benchmark duration
BENCHMARK_DURATION_IN_MINUTES = 3

def get_requests_with_time(input_requests: List[Tuple[str, int, int]],
                           high_request_rate: float,
                           mid_request_rate: float,
                           low_request_rate: float) -> List[Tuple[float, Tuple[str, int, int]]]:
    """
    Generates requests with associated times based on a custom request rate pattern:
    Long Low Rate -> Mid Rate -> Short High Rate -> Mid Rate -> Long Low Rate
    """
    requests_with_time = []
    current_time = 0.0
    total_duration = BENCHMARK_DURATION_IN_MINUTES * 60  # Total benchmark duration in seconds

    # Define the durations for each phase
    phase_durations = [
        total_duration / 5, # Phase 1: Low Rate 
        total_duration / 5,  # Phase 2: Mid Rate
        total_duration / 5,  # Phase 3: High Rate 
        total_duration / 5,  # Phase 4: Mid Rate 
        total_duration / 5  # Phase 5: Low Rate
    ]

    # Define the phases with their corresponding request rates
    phases = [
        (low_request_rate, phase_durations[0]),   # Phase 1: Low Rate
        (mid_request_rate, phase_durations[1]),   # Phase 2: Mid Rate
        (high_request_rate, phase_durations[2]),  # Phase 3: High Rate
        (mid_request_rate, phase_durations[3]),   # Phase 4: Mid Rate
        (low_request_rate, phase_durations[4]),   # Phase 5: Low Rate
    ]

    phase_index = 0
    current_request_rate, phase_duration = phases[phase_index]
    time_period_end = phase_duration

    for request in cycle(input_requests):
        # Generate inter-arrival time based on current request rate
        if current_request_rate > 0:
            interval = np.random.exponential(1.0 / current_request_rate)
        else:
            interval = float('inf')
        current_time += interval

        # Update request rate based on time
        if current_time > time_period_end and phase_index < len(phases) - 1:
            phase_index += 1
            current_request_rate, phase_duration = phases[phase_index]
            time_period_end += phase_duration  # Accumulate the durations for time thresholds

        if current_time > total_duration:
            break

        requests_with_time.append((current_time, request))

    return requests_with_time

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

def run(llm: LLM, requests: List[Tuple[str, int, int]],
        high_request_rate: float, mid_request_rate: float, low_request_rate: float,
        temperature: float) -> Tuple[dict, int, bool, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Runs the benchmark, processing requests with the given LLM."""
    requests_with_time = get_requests_with_time(requests, high_request_rate, mid_request_rate, low_request_rate)
    outputs: List[RequestOutput] = []
    result = {}
    token_latencies_over_time = []
    token_throughput_over_time = []  # New: To store token throughput over time
    requests_over_time = []

    interval_duration = 10.0  # Interval duration in seconds
    next_interval_time = interval_duration  # Time when the next interval ends
    tokens_in_interval = 0  # Tokens processed in the current interval

    start_time = time.perf_counter()

    total_processed_tokens = 0
    request_index = 0
    while True:
        current_time = time.perf_counter() - start_time

        # Add requests to the engine if their scheduled time has passed
        while request_index < len(requests_with_time) and requests_with_time[request_index][0] <= current_time:
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
            result[request_id] = [request_start_time]
            request_index += 1

        step_outputs = llm.llm_engine.step()
        for output in step_outputs:
            if len(output.outputs[0].token_ids) == 1 and len(result[output.request_id]) == 1:
                ttft = time.perf_counter() - start_time - \
                    result[output.request_id][0]
                result[output.request_id].append(ttft)

            if output.finished:
                current_time_in_run = time.perf_counter() - start_time
                e2e_latency = current_time_in_run - \
                    result[output.request_id][0]
                result[output.request_id].extend(
                    [e2e_latency, len(output.prompt_token_ids), len(output.outputs[0].token_ids)])
                outputs.append(output)
                num_tokens = len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                total_processed_tokens += num_tokens
                tokens_in_interval += num_tokens  # Update tokens in the current interval

                # Calculate token latency
                token_latency = e2e_latency / len(output.outputs[0].token_ids)
                token_latencies_over_time.append((current_time_in_run, token_latency))
        
                num_request = len(step_outputs)
                requests_over_time.append((time.perf_counter() - start_time, num_request))

        # Check if we've reached the end of the current interval
        if current_time >= next_interval_time:
            interval_throughput = tokens_in_interval / interval_duration
            token_throughput_over_time.append((next_interval_time, interval_throughput))
            tokens_in_interval = 0  # Reset token count for next interval
            next_interval_time += interval_duration
            print(f"Token Throughput at {next_interval_time}s: {interval_throughput:.3f} tokens/s")

        throughput = total_processed_tokens / \
            (time.perf_counter() - start_time)
        print(f"Throughput: {throughput:.3f} tokens/s")

        if not llm.llm_engine.has_unfinished_requests() and (current_time > BENCHMARK_DURATION_IN_MINUTES * 60):
            break

    # Remove request_id from result if not exist in outputs
    for request_id in list(result.keys()):
        if request_id not in [output.request_id for output in outputs]:
            del result[request_id]

    total_tokens = sum(prompt_len + output_len for _, _, _,
                       prompt_len, output_len in result.values())

    return result, total_tokens,  token_latencies_over_time, token_throughput_over_time, requests_over_time  # Return the throughput data

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
        ["Selective Validation", args.selective_validation],
        ["Drop Threshold", args.drop_threshold],
        ["Consolidated Attention", args.consolidated_attention],
        ["Dataset", args.dataset],
        ["High Request Rate", args.high_request_rate],
        ["Mid Request Rate", args.mid_request_rate],
        ["Low Request Rate", args.low_request_rate],
        ["Benchmark Duration (min)", args.benchmark_duration],
    ]
    print(tabulate(config_table))

    global BENCHMARK_DURATION_IN_MINUTES
    BENCHMARK_DURATION_IN_MINUTES = args.benchmark_duration

    llm = LLM(
        model=args.target_model,
        gpu_memory_utilization=0.85,
    )

    # Sample the requests
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    warmup(llm)

    # 0 indicates all requests
    requests = sample_requests(args.dataset, 0, tokenizer)
    # Run the benchmark
    start_time = time.perf_counter()
    result, total_tokens, token_latencies_over_time, token_throughput_over_time, requests_over_time = run(
        llm, requests, args.high_request_rate, args.mid_request_rate, args.low_request_rate, args.temperature)
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

    # Print all results in csv format
    # print("Request Throughput (reqs/s),Token Throughput (tokens/s),Token Latency (s/token),"
    #       "P50 TTFT (s),P99 TTFT (s),P50 TPOT (s/token),P99 TPOT (s/token),P50 Token Latency (s/token),P99 Token Latency (s/token),"
    #       "Preempt Flag")
    # print(f"result, {request_throughput:.3f}, {token_throughput:.3f}, {token_latency:.6f}, {p50_ttft:.6f}, {p99_ttft:.6f}, {p50_tpot:.6f}, {p99_tpot:.6f}, {p50_token_latency:.6f}, {p99_token_latency:.6f}, {preempt_flag}")

    file_name_prefix = f"AR"

    # Write token latencies over time to a CSV file
    with open(f'token_latencies_over_time_{file_name_prefix}.csv', 'w') as f:
        f.write('Time(s),Token_Latency(s/token)\n')
        for time_point, latency in token_latencies_over_time:
            f.write(f"{time_point},{latency}\n")

    # Write token throughput over time to a CSV file
    with open(f'token_throughput_over_time_{file_name_prefix}.csv', 'w') as f:
        f.write('Time(s),Token_Throughput(tokens/s)\n')
        for time_point, throughput in token_throughput_over_time:
            f.write(f"{time_point},{throughput}\n")

    with open(f'requests_over_time_{file_name_prefix}.csv', 'w') as f:
        f.write('Time(s),Num_Requests\n')
        for time_point, num_requests in requests_over_time:
            f.write(f"{time_point},{num_requests}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=0)
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
    parser.add_argument('--benchmark-duration', type=int, default=5,
                        help='Benchmark duration in minutes.')
    parser.add_argument('--low-request-rate', type=float, default=6,
                        help='Low request rate for changing arrival pattern.')
    parser.add_argument('--mid-request-rate', type=float, default=12,
                        help='Mid request rate for changing arrival pattern.')
    parser.add_argument('--high-request-rate', type=float, default=24,
                        help='High request rate for changing arrival pattern.')

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    assert args.dataset is not None

    main(args)
