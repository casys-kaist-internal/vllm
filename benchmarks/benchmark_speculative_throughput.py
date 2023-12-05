"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams, SpSLLM
from vllm.transformers_utils.tokenizer import get_tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

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
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_base(
    requests: List[Tuple[str, int, int]],
    target_model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
) -> float:
    llm = LLM(
        model=target_model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    outputs = llm._run_engine(use_tqdm=True)
    end = time.time()

    print("Prompt: ", outputs[0].prompt,
          "\nOutput: ", outputs[0].outputs[0].text)

    return end - start


def run_sps(
    requests: List[Tuple[str, int, int]],
    target_model: str,
    draft_model: str,
    draft_size: int,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: int,
    seed: int,
) -> float:

    llm = SpSLLM(
        target_model=target_model,
        draft_model=draft_model,
        draft_size=draft_size,
        tokenizer=tokenizer,
        target_tensor_parallel_size=tensor_parallel_size,
        draft_tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    outputs = llm._run_engine(use_tqdm=True)
    end = time.time()

    print("Prompt: ", outputs[0].prompt,
          "\nOutput: ", outputs[0].outputs[0].text)

    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.engine == "base":
        elapsed_time = run_base(
            requests, args.target_model, args.tokenizer, args.tensor_parallel_size, args.seed)
    elif args.engine == "sps":
        elapsed_time = run_sps(
            requests, args.target_model, args.draft_model, args.draft_size, args.tokenizer, args.tensor_parallel_size, args.seed)
    else:
        raise ValueError(f"Unknown engine: {args.engine}")
    total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--engine", type=str, choices=["base", "sps"],
                        default="base")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--target-model", type=str,
                        default="facebook/opt-6.7B")
    parser.add_argument("--draft-model", type=str,
                        default="facebook/opt-125m")
    parser.add_argument('--draft-size', type=int, default=2)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    main(args)
