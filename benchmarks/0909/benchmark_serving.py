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
BENCHMARK_DURATION_IN_MINUTES = 5

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


def run(llm: SpecDecodeLLM, requests: List[Tuple[str, int, int]], request_rate: float, temperature: float) -> Tuple[dict, int, bool]:
    """Runs the benchmark, processing requests with the given LLM."""
    requests_with_time = get_requests_with_time(requests, request_rate)
    outputs: List[RequestOutput] = []
    result = {}

    start_time = time.perf_counter()
    llm.llm_engine.reset_total_tokens()
    start_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()

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
                prompt=prompt, prompt_token_ids=None, sampling_params=sampling_params)
            result[request_id] = [request_start_time]
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

    llm.llm_engine.abort_all_requests()
    llm.llm_engine.worker_executor.shutdown()

    end_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()
    assert start_num_free_gpu_blocks == end_num_free_gpu_blocks

    return result, total_tokens, llm.llm_engine.scheduler.preempt_flag


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
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    # 0 indicates all requests
    requests = sample_requests(args.dataset, 0, tokenizer)

    # Run the benchmark
    start_time = time.perf_counter()
    result, total_tokens, preempt_flag = run(
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

    # Print all results in csv format
    print("Request Throughput (reqs/s),Token Throughput (tokens/s),Token Latency (s/token),"
          "P50 TTFT (s),P99 TTFT (s),P50 TPOT (s/token),P99 TPOT (s/token),P50 Token Latency (s/token),P99 Token Latency (s/token),"
          "Preempt Flag")
    print(f"result, {request_throughput:.3f}, {token_throughput:.3f}, {token_latency:.6f}, {p50_ttft:.6f}, {p99_ttft:.6f}, {p50_tpot:.6f}, {p99_tpot:.6f}, {p50_token_latency:.6f}, {p99_token_latency:.6f}, {preempt_flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None,
                        help="Input prompt length for each request.")
    parser.add_argument("--output-len", type=int, default=None,
                        help="Output length for each request. Overrides the output length from the dataset.")
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

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    if args.dataset is None or args.dataset == "dummy":
        args.dataset = "dummy"
        assert args.input_len is not None
        assert args.output_len is not None
        assert args.emulate_accept_prob is not None
    else:
        assert args.input_len is None

    main(args)
