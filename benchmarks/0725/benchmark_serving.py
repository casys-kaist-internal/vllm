"""Benchmark offline inference throughput."""
import argparse
import random
import time
import os
import gc
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Tuple
from transformers import (AutoTokenizer)
from dataset import sample_requests
from tabulate import tabulate
import torch
import matplotlib.pyplot as plt
from itertools import cycle

from vllm import LLM, SpecDecodeLLM, SamplingParams
from vllm.outputs import RequestOutput


DOWNLOAD_DIR = '/mnt/sda/download'
BENCHMARK_DURATION_IN_MINUTES = 3

gc.disable()


def get_requests_with_time(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> List[Tuple[float, Tuple[str, int, int]]]:
    input_requests = iter(input_requests)
    requests_with_time = []
    current_time = 0.0

    for request in cycle(input_requests):
        requests_with_time.append((current_time, request))

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # Accumulate the interval to get the next request time.
        current_time += interval

        if current_time > (BENCHMARK_DURATION_IN_MINUTES + 1) * 60:
            break

    return requests_with_time


def run(
        llm: SpecDecodeLLM,
        requests: List[Tuple[str, int, int]],
        request_rate: float,
        temperature: float) -> Tuple[dict, int]:
    requests_with_time = get_requests_with_time(requests, request_rate)

    outputs: List[RequestOutput] = []
    start_time = time.perf_counter()
    # request_id -> [request_start_time, ttft, e2e_latency]
    result = {}

    llm.llm_engine.reset_total_tokens()
    start_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()

    request_index = 0
    while True:
        current_time = time.perf_counter() - start_time

        if current_time > BENCHMARK_DURATION_IN_MINUTES * 60:
            break

        # Add requests to the engine if their scheduled time has passed
        while requests_with_time[request_index][0] <= current_time:
            request_start_time, (prompt, prompt_len,
                                 output_len) = requests_with_time[request_index]
            if temperature == -1:  # Random temperature
                # random from 0 0.25 0.5 0.75 1
                random_temperautre = random.choice([0, 0.25, 0.5, 0.75, 1])
                sampling_params = SamplingParams(
                    n=1,
                    temperature=random_temperautre,
                    top_p=1.0,
                    use_beam_search=False,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
            else:
                sampling_params = SamplingParams(
                    n=1,
                    temperature=temperature,
                    top_p=1.0,
                    use_beam_search=False,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
            request_id = llm._add_request(prompt=prompt,
                                          prompt_token_ids=None,
                                          sampling_params=sampling_params)
            # Record start time
            result[request_id] = [request_start_time]
            request_index += 1

        start_time_ = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        torch.cuda.synchronize()
        elasped_time = time.perf_counter() - start_time_
        elasped_time_ms = elasped_time * 1000
        print(f"Budget, {elasped_time_ms:.3f}")

        for output in step_outputs:
            if len(output.outputs[0].token_ids) == 1 and len(result[output.request_id]) == 1:
                ttft = (time.perf_counter() - start_time) - \
                    result[output.request_id][0]
                result[output.request_id].append(ttft)

            if output.finished:
                if len(result[output.request_id]) == 1:
                    ttft = (time.perf_counter() - start_time) - \
                        result[output.request_id][0]
                    result[output.request_id].append(ttft)
                e2e_latency = (time.perf_counter() - start_time) - \
                    result[output.request_id][0]
                result[output.request_id].append(e2e_latency)
                result[output.request_id].append(len(output.prompt_token_ids))
                result[output.request_id].append(
                    len(output.outputs[0].token_ids))
                outputs.append(output)

        throughput = len(outputs) / (time.perf_counter() - start_time)
        print(f"!!! Throughput: {throughput:.3f} reqs/s")

    prefill_tokens, decode_tokens = llm.llm_engine.get_total_tokens()
    total_tokens = prefill_tokens + decode_tokens

    print(f"prefill percentage: {prefill_tokens / total_tokens}")
    print(f"decode percentage: {decode_tokens / total_tokens}")

    # remove request_id from result if not exist in outputs
    for request_id in list(result.keys()):
        if request_id not in [output.request_id for output in outputs]:
            del result[request_id]

    llm.llm_engine.abort_all_requests()
    llm.llm_engine.worker_executor.shutdown()

    end_num_free_gpu_blocks = llm.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()

    assert start_num_free_gpu_blocks == end_num_free_gpu_blocks

    return result, total_tokens, llm.llm_engine.scheduler.preempt_flag


def analyze_results(request_rate: int, result: dict):
    ttfts = []  # Time to first token
    tpots = []  # Time per output tokens
    tpts = []  # Time per tokens

    for request_id, values in result.items():
        request_start_time, ttft, e2e_latency, prompt_len, output_len = values
        ttfts.append(ttft)
        tpot = (e2e_latency - ttft) / (output_len - 1)
        tpots.append(tpot)
        tpt = e2e_latency / (prompt_len + output_len)
        tpts.append(tpt)

    # Compute the latency statistics.
    ttfts = np.array(ttfts)
    tpots = np.array(tpots)
    tpts = np.array(tpts)

    return ttfts, tpots, tpts


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    table = [["target_model", args.target_model],
             ["draft_model", args.draft_model],
             ["draft_size", args.draft_size],
             ["temperature", args.temperature]]

    print(tabulate(table))

    table = [["colocate", args.colocate],
             ["prefill_schedule_mode", args.prefill_schedule_mode],
             ["budget_token", args.budget_token],
             ["budget_seq", args.budget_seq],
             ["drop_threshold", args.drop_threshold],
             ["target_attention", args.target_attention],]

    print(tabulate(table))

    table = [["dataset", args.dataset],
             ["request_rate", args.request_rate],
             ["emulate_accept_prob", args.emulate_accept_prob],
             ["input_len", args.input_len],
             ["output_len", args.output_len],]

    print(tabulate(table))

    llm = SpecDecodeLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        colocate=args.colocate,
        prefill_schedule_mode=args.prefill_schedule_mode,
        target_attention=args.target_attention,
        drop_threshold=args.drop_threshold,
        emulate_accept_prob=args.emulate_accept_prob,
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
        # disable_log_stats=False,
    )

    def warmup():
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
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

    # warmup()

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    requests = sample_requests(args.dataset, 0, tokenizer,
                               fixed_input_len=args.input_len,
                               fixed_output_len=args.output_len)  # 0 indicates all requests

    all_ttfts = []
    all_tpots = []
    all_tpts = []
    throughputs = []
    latencies = []

    # print(f"Running benchmark with request rate: {request_rate} reqs/s")
    start_time = time.perf_counter()
    result, total_tokens, preempt_flag = run(
        llm, requests, args.request_rate, args.temperature)
    elapsed_time = time.perf_counter() - start_time

    ttfts, tpots, tpts = analyze_results(args.request_rate, result)

    print(total_tokens, elapsed_time)

    throughput = total_tokens / elapsed_time
    request_per_second = len(result) / elapsed_time
    throughputs.append(throughput)
    latencies.append(np.mean(tpts))

    all_ttfts.append(ttfts)
    all_tpots.append(tpots)
    all_tpts.append(tpts)

    p50_ttft = np.percentile(ttfts, 50)
    p99_ttft = np.percentile(ttfts, 99)
    p50_tpot = np.percentile(tpots, 50)
    p99_tpot = np.percentile(tpots, 99)
    p50_tpt = np.percentile(tpts, 50)
    p99_tpt = np.percentile(tpts, 99)

    print(f"result, {p50_ttft:.3f}, {p99_ttft:.3f}, {p50_tpot:.3f}, {p99_tpot:.3f}, {p50_tpt:.3f}, {p99_tpt:.3f}, {throughput:.3f}, {request_per_second:.3f}, {np.mean(tpts):.3f}, {preempt_flag}")

    # plot_metrics(request_rates, all_ttfts, all_tpots, all_tpts)
    # plot_latency_throughput(request_rates, latencies, throughputs)


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
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="Temperature for sampling. -1 for random temperature.")
    parser.add_argument('--colocate',
                        '-c',
                        action='store_true')
    parser.add_argument('--prefill-schedule-mode',
                        '-psm',
                        choices=['prioritize_prefill',
                                 'full_prefill', 'chunked_prefill', 'chunked_prefill_demote_draft'],
                        default='full_prefill')
    parser.add_argument("--target-attention",
                        action="store_true",
                        help="Use target attention.")
    parser.add_argument("--drop-threshold",
                        '-dt',
                        type=float,
                        default=0,
                        help="Threshold for dropping token")
    parser.add_argument('--budget-token',
                        type=int,
                        default=2048,
                        help='Maximum number of tokens for each batch.')
    parser.add_argument('--budget-seq',
                        type=int,
                        default=128,
                        help='Maximum number of sequences for each request.')
    parser.add_argument("--emulate-accept-prob",
                        type=float,
                        default=None,
                        help="Emulate accept probability.")
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
                        default=4,
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
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
