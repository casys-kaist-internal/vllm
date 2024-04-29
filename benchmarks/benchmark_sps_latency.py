"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import time
import random
from typing import List, Optional, Tuple
from datasets import load_dataset

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)
from vllm import LLM, SpSLLM, SamplingParams

download_dir = '/home/sjchoi/workspace/models'


def load_gsm8k(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('gsm8k', 'main')['train']

    # Tokenize the prompts and completions.
    prompts = [data['question'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data['answer'] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # random sort dataset
    random.shuffle(filtered_dataset)

    return filtered_dataset

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase
) -> List[Tuple[str, int, int]]:

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = filtered_dataset[:num_requests]
    # sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def main(args: argparse.Namespace):
    global download_dir
    print(args)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    if args.engine == "base":
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
    elif args.engine == "sps":
        llm = SpSLLM(
            target_model=args.target_model,
            draft_model=args.draft_model,
            draft_size=args.draft_size,
            tile_size=args.tile_size,
            use_dynamic_draft_size=args.dynamic_draft,
            use_tile_size_constraint=False,
            use_lazy_draft_kv_cache=True,
            use_target_attention=args.use_target_attention,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            download_dir=download_dir
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)

    if args.dataset =="gsm8k":
        requests = load_gsm8k(tokenizer)
    else:
        dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion():
        output = llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                              sampling_params=sampling_params,
                              use_tqdm=False)
        # for prompt, _, output_len in requests[:args.batch_size]:
        #     print(prompt)
        #     print(output_len)
        #     sampling_params = SamplingParams(
        #         n=1,
        #         temperature=args.temperature,
        #         top_p=1.0,
        #         use_beam_search=args.use_beam_search,
        #         ignore_eos=True,
        #         max_tokens=512,
        #     )
        #     llm._add_request(
        #         prompt=prompt,
        #         prompt_token_ids=None,
        #         sampling_params=sampling_params,
        #     )
        # start_time = time.perf_counter()
        # llm._run_engine(use_tqdm=False)
        # end_time = time.perf_counter()
        # latency = end_time - start_time
        return latency

    # print("Warming up...")
    # run_to_completion()

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion())
    print(f'DECODE Avg latency: {np.mean(latencies)} seconds')


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
                        default="daryl149/llama-2-7b-chat-hf")
    # default="EleutherAI/pythia-6.9b")
    # default="bigscience/bloom-7b1")
    # default="meta-llama/Llama-2-7b-chat-hf")
    # default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str,
                        default='Felladrin/Llama-68M-Chat-v1')
    # default="facebook/opt-350m")
    # default="EleutherAI/pythia-14m")
    # default="EleutherAI/pythia-31m")
    # default="EleutherAI/pythia-70m")
    # default="EleutherAI/pythia-410m")
    # default="EleutherAI/pythia-160m")
    # default='bigscience/bloomz-560m')
    # default='bigscience/bloomz-560m')
    # default='Felladrin/Llama-68M-Chat-v1')
    # default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--dynamic-draft', action='store_true')
    parser.add_argument('--use-tile-size-constraint', action='store_true')
    parser.add_argument('--use-lazy-draft-kv-cache', action='store_true')
    parser.add_argument('--use-target-attention',
                        action='store_true')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=1024)
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
