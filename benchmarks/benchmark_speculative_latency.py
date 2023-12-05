"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import SpSLLM, SamplingParams


def main(args: argparse.Namespace):

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = SpSLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        tokenizer=args.tokenizer,
        target_tensor_parallel_size=args.target_tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.9,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    dummy_prompt_token_ids = ([[0] * args.input_len]) * args.batch_size

    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        output = llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                              sampling_params=sampling_params,
                              use_tqdm=False)  # 여기서부터 시작

        end_time = time.time()
        latency = end_time - start_time
        for i in range(args.batch_size):
            assert output[i].outputs[0].finish_reason == "length"

        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(latencies)
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str,
                        default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--target-tensor-parallel-size',
                        '-target-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=1,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)
