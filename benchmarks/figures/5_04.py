"""Benchmark the latency of processing a single batch of requests."""
import sys
import argparse
import json
import time
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)
from vllm import LLM, SpSLLM, SamplingParams
from datasets import load_dataset

download_dir = '/mnt/sda/models'
MAX_NUM_SEQUENCE = 8000


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
    # random.shuffle(filtered_dataset)

    return filtered_dataset


def load_humaneval(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('openai_humaneval')['test']

    # Tokenize the prompts and completions.
    prompts = [data['prompt'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data['canonical_solution'] for data in dataset]
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
    # random.shuffle(filtered_dataset)

    return filtered_dataset


def load_alpaca(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('tatsu-lab/alpaca')['train']

    # Tokenize the prompts and completions.
    prompts = [data['instruction'] + data['input'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data['output'] for data in dataset]
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
    # random.shuffle(filtered_dataset)

    return filtered_dataset


def load_mt_bench(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('philschmid/mt-bench')['train']
    prompts = [data['turns'][0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    completions = []
    # Open the output of gpt-4 mt-bench file to get completion token ids
    with open('gpt-4.jsonl', 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Parse the JSON data from each line
            json_object = json.loads(line)
            completions.append(json_object['choices'][0]['turns'][0])
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
    # random.shuffle(filtered_dataset)

    return filtered_dataset


def load_sharegpt(tokenizer: PreTrainedTokenizerBase):
    with open('/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
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
    # random.shuffle(filtered_dataset)

    return filtered_dataset

def load_apps(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('codeparrot/apps')['train']

    # Tokenize the prompts and completions.
    prompts = [data['question'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data['solutions'] for data in dataset]
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
    # random.shuffle(filtered_dataset)

    return filtered_dataset


def warmup(llm):
    dummy_prompt_token_ids = [[0] * 32] * 32
    dummy_sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=48,
    )
    llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                 sampling_params=dummy_sampling_params,
                 use_tqdm=False)


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
            use_tile_size_constraint=args.use_tile_size,
            use_lazy_draft_kv_cache=True,
            use_target_attention=args.use_target_attention,
            target_draft_latency_ratio=args.target_draft_latency_ratio,
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

    if args.dataset == "gsm8k":
        requests = load_gsm8k(tokenizer)
    elif args.dataset == "humaneval":
        requests = load_humaneval(tokenizer)
    elif args.dataset == "alpaca":
        requests = load_alpaca(tokenizer)
    elif args.dataset == "mt-bench":
        requests = load_mt_bench(tokenizer)
    elif args.dataset == "sharegpt":
        requests = load_sharegpt(tokenizer)
    elif args.dataset == "apps":
        requests = load_apps(tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Warmup
    # warmup(llm)

    # Profile 
    # llm._run_profile()

    # latencies = []
    throughputs = []
    # output_lens = []
    for i in range(0, min(len(requests), MAX_NUM_SEQUENCE), args.batch_size):
        sampled_requests = requests[i: min(i + args.batch_size, MAX_NUM_SEQUENCE)]

        # print(sampled_requests)
        for prompt, _, output_len in sampled_requests:
            # print(prompt)

            if args.random_temp:
                sampling_params = SamplingParams(
                    n=1,
                    temperature=random.uniform(0.0, 1.0),
                    frequency_penalty=args.frequency_penalty,
                    top_p=1.0,
                    use_beam_search=False,
                    ignore_eos=True,
                    max_tokens=512,
                )
            else:
                sampling_params = SamplingParams(
                    n=1,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    top_p=1.0,
                    use_beam_search=False,
                    ignore_eos=True,
                    max_tokens=512,
                )
            # FIXME(woosuk): Do not use internal method.
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
            )
        
        torch.cuda.synchronize()
        start_time = time.monotonic()
        outputs = llm._run_engine(use_tqdm=True)
        torch.cuda.synchronize()
        end_time = time.monotonic()

        # print(outputs)
        total_tokens = 0
        output_tokens = 0
        for idx, output in enumerate(outputs):
            output_tokens += len(output.outputs[0].token_ids)
            total_tokens += (sampled_requests[idx][1] + len(output.outputs[0].token_ids))
            # print("-" * 80)
            # print(f"{idx} Prompt: {sampled_requests[idx][0]}")
            # print(f"{idx} Output: {output.outputs[0].text}")
        throughputs.append(total_tokens / (end_time - start_time))
        print(f"throughput, {total_tokens / (end_time - start_time):.3f}, {output_tokens / (end_time - start_time):.3f}")
        # print(f"latency: {end_time - start_time:.3f}")
        # print(f"Generation latency: {generation_latency:.3f} seconds")
        # print(f"Output length: {output_len}")

        # per request per output token latency
    #     latency = generation_latency / np.mean(output_len)
    #     throughput = np.sum(output_len) / generation_latency
    #     latencies.append(latency)
    #     throughputs.append(throughput)
    #     output_lens.append(np.mean(output_len))
    print("throughput: ", np.mean(throughputs))

    # print(
    #     f"result, {np.mean(latencies):.6f}, {np.mean(throughputs):.6f}, {np.mean(output_lens):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument("--engine", type=str, choices=["base", "sps"],
                        default="base")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "humaneval",
                                 "alpaca", "mt-bench", "sharegpt", "apps"],
                        help="Dataset to use.")
    parser.add_argument('--target-model', type=str, 
                        # default='EleutherAI/pythia-12b')
                        default='facebook/opt-6.7b')
                        # default='bigscience/bloom-7b1')
                        # default='daryl149/llama-2-7b-chat-hf')
                        # default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, 
                        # default='EleutherAI/pythia-410m')
                        # default='bigscience/bloomz-560m')
                        # default='Felladrin/Llama-68M-Chat-v1')
                        default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--dynamic-draft', action='store_true')
    parser.add_argument('--use-tile-size', action='store_true')
    parser.add_argument('--use-lazy-draft-kv-cache', action='store_true')
    parser.add_argument('--use-target-attention',
                        action='store_true')
    parser.add_argument('--target-draft-latency-ratio', 
                        '-c',
                        type=float, default=0.2)
    parser.add_argument('--frequency-penalty', type=float, default=0.0)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--temperature',
                        '-t',
                        type=float,
                        default=0.5,
                        help='Sampling temperature.')
    parser.add_argument('--random-temp', action='store_true')
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
