"""Benchmark the latency of processing a single batch of requests."""
import sys
import argparse
import json
import time
import random
from typing import List, Optional, Tuple
import uuid

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)
from vllm import LLM, SpSLLM, SamplingParams
from datasets import load_dataset

download_dir = '/data/models'
dataset_download_dir = '/data/datasets'


def load_requests_from_file(file_path):
    with open(file_path, 'r') as file:
        requests = json.load(file)
    return requests

def load_dataset_with_reject_pos(dataset, target_model, draft_model, temperature):
    cleaned_target_model = target_model.split('/')[1]
    cleaned_draft_model = draft_model.split('/')[1]
    cleaned_temperature = str(temperature).replace('.', '_')

    return load_requests_from_file(f'{dataset_download_dir}/{dataset}_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}.json')


def load_all_datasets_with_reject_pos(target_model, draft_model, temperature):
    result = []

    for dataset in ["gsm8k", "humaneval", "alpaca", "mt-bench", "sharegpt", "apps"]:
        result.extend(load_dataset_with_reject_pos(dataset, target_model, draft_model, temperature))

    # shuffle all datasets
    random.shuffle(result)

    return result

def main(args: argparse.Namespace):
    global download_dir
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'all':
        requests = load_all_datasets_with_reject_pos(args.target_model, args.draft_model, args.temperature)
    else:
        requests = load_dataset_with_reject_pos(args.dataset, args.target_model, args.draft_model, args.temperature)
    
    print(f"Total requests loaded: {len(requests)}")  # Announce total number of requests
    llm = SpSLLM(
            target_model=args.target_model,
            draft_model=args.draft_model,
            draft_size=args.draft_size,
            tile_size=args.tile_size,
            use_dynamic_draft_size=args.dynamic_draft,
            use_tile_size_constraint=args.use_tile_size,
            use_lazy_draft_kv_cache=True,
            use_target_attention=args.use_target_attention,
            emulate_accept_reject=True,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            download_dir=download_dir
        )
    
    if isinstance(llm, SpSLLM):
        llm.llm_engine.workers[0].draft_optimizer.reset()

    save_dict = {}
    batch = []   
    batch_size = 32
    result = []
    completed_requests = 0  # Counter for tracking completed requests

    for request in requests:
        request_id = request['request_id']
        prompt = request['prompt']
        output_len = 512
        reject_pos = request['reject_pos']

        save_dict[request_id] = request

        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=output_len,
        )

        # Collect requests into the batch
        batch.append((request_id, prompt, None, sampling_params, reject_pos))
        
        # Check if batch is full
        if len(batch) == batch_size:
            # Process the full batch
            for req_id, prmpt, prmpt_ids, s_params, reject_pos in batch:
                llm._add_request(
                    request_id=req_id,
                    prompt=prmpt,
                    prompt_token_ids=prmpt_ids,
                    sampling_params=s_params,
                    reject_pos=reject_pos
                )
            outputs = llm._run_engine(use_tqdm=False)
            completed_requests += len(batch)
            print(f"Processed {completed_requests}/{len(requests)} requests.")
            result.extend(outputs)
            batch = []  # Reset the batch after processing
    
    # Process any remaining requests in the last partial batch
    if batch:
        for req_id, prmpt, prmpt_ids, s_params, reject_pos in batch:
            llm._add_request(
                request_id=req_id,
                prompt=prmpt,
                prompt_token_ids=prmpt_ids,
                sampling_params=s_params,
                reject_pos=reject_pos
            )
        outputs = llm._run_engine(use_tqdm=False)
        result.extend(outputs)

    # total_result = []
    # for output in result:
    #     request_id = output.request_id
    #     reject_pos = output.outputs[0].reject_pos
    #     prompt_len = save_dict[request_id]['prompt_len']
    #     output_len = save_dict[request_id]['output_len']
    #     prompt = save_dict[request_id]['prompt']

    #     result = {
    #         "request_id": request_id,
    #         "prompt": prompt,
    #         "reject_pos": reject_pos,
    #         "prompt_len": prompt_len,
    #         "output_len": output_len
    #     }

    #     total_result.append(result)

    # cleaned_target_model = args.target_model.split('/')[1]
    # cleaned_draft_model = args.draft_model.split('/')[1]
    # cleaned_temperature = str(args.temperature).replace('.', '_')

    # with open(f'{dataset_download_dir}/{args.dataset}_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}.json', 'w') as file:
    #     json.dump(total_result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--engine", type=str, choices=["base", "sps"],
                        default="sps")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "humaneval",
                                 "alpaca", "mt-bench", "sharegpt", "apps", "all"],
                        help="Dataset to use.")
    parser.add_argument('--target-model', type=str,
                        # default='EleutherAI/pythia-6.9b') 
                        # default='EleutherAI/pythia-12b')
                        default='facebook/opt-6.7b')
                        # default='bigscience/bloom-7b1')
                        # default='daryl149/llama-2-7b-chat-hf')
                        # default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, 
                        # default='EleutherAI/pythia-14m')
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
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature',
                        '-t',
                        type=float,
                        default=0.75,
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
