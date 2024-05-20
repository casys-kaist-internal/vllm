import os
import argparse
import json
import time
import random
from typing import List, Optional, Tuple
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import SpSLLM, SamplingParams
from datasets import load_dataset

# Change to the desired GPU ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# current date 
current_time = time.strftime("%Y%m%d-%H%M%S")

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'
OUTPUT_DIR = f'/mnt/sda/results/{current_time}'
PREDICTOR_PATH = 'predictor_10000'
MAX_NUM_SEQUENCE = 10000

# Test cases
STATIC = False
STATIC_TILE = False
DYNAMIC = False
DYNAMIC_TILE = False


def load_gsm8k(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('gsm8k', 'main', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'question', 'answer', tokenizer)


def load_humaneval(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('openai_humaneval', cache_dir=DOWNLOAD_DIR)['test']
    return process_dataset(dataset, 'prompt', 'canonical_solution', tokenizer)


def load_alpaca(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('tatsu-lab/alpaca', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'instruction', 'output', tokenizer, input_key='input')


def load_mt_bench(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('philschmid/mt-bench',
                           cache_dir=DOWNLOAD_DIR)['train']
    prompts = [data['turns'][0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    completions = []
    with open('gpt-4.jsonl', 'r') as file:
        for line in file:
            json_object = json.loads(line)
            completions.append(json_object['choices'][0]['turns'][0])
    completion_token_ids = tokenizer(completions).input_ids

    return filter_and_process(prompts, prompt_token_ids, completion_token_ids)


def load_sharegpt(tokenizer: PreTrainedTokenizerBase):
    with open(f'{DOWNLOAD_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
        dataset = json.load(f)

    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    return filter_and_process(prompts, prompt_token_ids, completion_token_ids)


def load_apps(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('codeparrot/apps', cache_dir=DOWNLOAD_DIR)['train']
    return process_dataset(dataset, 'question', 'solutions', tokenizer)


def process_dataset(dataset, prompt_key, completion_key, tokenizer, input_key=None):
    prompts = [data[prompt_key] +
               (data[input_key] if input_key else '') for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data[completion_key] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    return filter_and_process(prompts, prompt_token_ids, completion_token_ids)


def filter_and_process(prompts, prompt_token_ids, completion_token_ids):
    tokenized_dataset = [(prompts[i], prompt_token_ids[i], len(
        completion_token_ids[i])) for i in range(len(prompts))]
    filtered_dataset = [
        (prompt, len(prompt_ids), output_len)
        for prompt, prompt_ids, output_len in tokenized_dataset
        if 4 <= len(prompt_ids) <= 1024 and 4 <= output_len <= 2048 and len(prompt_ids) + output_len <= 2048
    ]
    return filtered_dataset


def load_all_datasets(tokenizer: PreTrainedTokenizerBase):
    datasets = [
        load_gsm8k(tokenizer),
        load_humaneval(tokenizer),
        load_alpaca(tokenizer),
        load_mt_bench(tokenizer),
        load_sharegpt(tokenizer),
        load_apps(tokenizer)
    ]
    combined_dataset = [item for sublist in datasets for item in sublist]
    random.shuffle(combined_dataset)
    return combined_dataset


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
                 sampling_params=dummy_sampling_params, use_tqdm=False)


def save_results_to_csv(throughputs, args):
    print("Saving results to CSV...")
    # Calculate the mean throughput for each configuration
    mean_throughputs = {}
    for key in throughputs.keys():
        mean_throughputs[key] = np.mean(throughputs[key])

    # print the results with pprint 
    pprint(mean_throughputs)

    keys = list(throughputs.keys())
    header = ", ".join(keys)

    max_length = max(len(value) for value in throughputs.values())
    rows = [
        ", ".join(f"{throughputs[key][i]:.3f}" if i < len(
            throughputs[key]) else "" for key in keys)
        for i in range(max_length)
    ]

    csv_output = header + "\n" + "\n".join(rows)

    cleaned_target_model = args.target_model.split("/")[-1]
    cleaned_draft_model = args.draft_model.split("/")[-1]
    cleaned_temperature = str(args.temperature).replace(".", "_")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_result_file_name = f"total_results_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}.csv"

    with open(os.path.join(OUTPUT_DIR, total_result_file_name), "w") as f:
        f.write(csv_output)

    mean_result_file_name = f"mean_results_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}.csv"
    header = ", ".join(keys)
    mean_row = ", ".join(f"{mean_throughputs[key]:.3f}" for key in keys)
    csv_output = header + "\n" + mean_row

    with open(os.path.join(OUTPUT_DIR, mean_result_file_name), "w") as f:
        f.write(csv_output)

def benchmark(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    llm = SpSLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        use_dynamic_draft_size=args.use_dynamic_draft,
        use_tile_constraint=args.use_tile_constraint,
        use_lazy_draft_kv_cache=True,
        use_target_attention=args.use_target_attention,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        download_dir=DOWNLOAD_DIR
    )

    if args.dataset == "all":
        requests = load_all_datasets(tokenizer)
    else:
        load_function = {
            "gsm8k": load_gsm8k,
            "humaneval": load_humaneval,
            "alpaca": load_alpaca,
            "mt-bench": load_mt_bench,
            "sharegpt": load_sharegpt,
            "apps": load_apps
        }.get(args.dataset)
        if load_function:
            requests = load_function(tokenizer)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    warmup(llm)

    cleaned_target_model = args.target_model.split("/")[-1]
    cleaned_draft_model = args.draft_model.split("/")[-1]
    cleaned_temperature = str(args.temperature).replace(".", "_")
    predictor_path = f"{PREDICTOR_PATH}/predictor_degree{args.predictor_degree}_{args.predictor_agg_type}_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}.csv"
    llm.llm_engine.workers[0].draft_optimizer.initialize(predictor_path)

    if llm.llm_engine.workers[0].draft_optimizer.retrain:
        pretrain(llm, requests, args)
        llm.llm_engine.workers[0].draft_optimizer.save_predictor()

    throughputs = {f"static_{i}": [] for i in range(8)}
    throughputs.update({"static_tile": [], "dynamic": [], "dynamic_tile": []})

    # Adjust the requests to be multiples of the batch size
    adjusted_max_num_sequence = (
        min(MAX_NUM_SEQUENCE, len(requests)) // args.batch_size) * args.batch_size

    for i in tqdm(range(0, adjusted_max_num_sequence, args.batch_size)):
        sampled_requests = requests[i:i + args.batch_size]

        if STATIC:
            benchmark_static_draft(llm, sampled_requests, throughputs, args)

        if STATIC_TILE:
            benchmark_static_tile(llm, sampled_requests, throughputs, args)

        if DYNAMIC:
            benchmark_dynamic_draft(llm, sampled_requests, throughputs, args)

        if DYNAMIC_TILE:
            benchmark_dynamic_tile(llm, sampled_requests, throughputs, args)

    save_results_to_csv(throughputs, args)

def pretrain(llm, requests, args):
    print("Pretraining the predictor...")
    # Adjust the requests to be multiples of the batch size
    adjusted_max_num_sequence = (
        min(MAX_NUM_SEQUENCE, len(requests)) // args.batch_size) * args.batch_size
    
    for i in tqdm(range(0, adjusted_max_num_sequence, args.batch_size)):
        sampled_requests = requests[i:i + args.batch_size]
        for prompt, _, output_len in sampled_requests:
            sampling_params = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=512,  # currently fixed to 512 but can be changed to output_len
            )
            llm._add_request(prompt=prompt, prompt_token_ids=None,
                             sampling_params=sampling_params)

        llm._run_engine(use_tqdm=False)

def benchmark_static_draft(llm, sampled_requests, throughputs, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "none"

    for draft_size in range(8):
        llm.llm_engine.sps_config.draft_size = draft_size
        benchmark_requests(llm, sampled_requests,
                           throughputs[f"static_{draft_size}"], args)


def benchmark_static_tile(llm, sampled_requests, throughputs, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "cut-128"
    llm.llm_engine.sps_config.draft_size = 7

    benchmark_requests(llm, sampled_requests, throughputs["static_tile"], args)


def benchmark_dynamic_draft(llm, sampled_requests, throughputs, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "none"

    benchmark_requests(llm, sampled_requests, throughputs["dynamic"], args)


def benchmark_dynamic_tile(llm, sampled_requests, throughputs, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "cut-128"

    benchmark_requests(llm, sampled_requests,
                       throughputs["dynamic_tile"], args)


def benchmark_requests(llm: SpSLLM, sampled_requests, throughput_list, args):
    for prompt, _, output_len in sampled_requests:
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=512,  # currently fixed to 512 but can be changed to output_len
        )
        llm._add_request(prompt=prompt, prompt_token_ids=None,
                         sampling_params=sampling_params)

    torch.cuda.synchronize()
    start_time = time.monotonic()
    outputs = llm._run_engine(use_tqdm=False)
    torch.cuda.synchronize()
    end_time = time.monotonic()

    total_tokens = sum(len(output.outputs[0].token_ids) +
                       sampled_requests[idx][1] for idx, output in enumerate(outputs))
    throughput_list.append(total_tokens / (end_time - start_time))


def main(args: argparse.Namespace):
    benchmark(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of requests till completion.')
    parser.add_argument("--dataset", type=str, default="humaneval",
                        choices=["gsm8k", "humaneval", "alpaca",
                                 "mt-bench", "sharegpt", "apps", "all"],
                        help="Dataset to use.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--use-dynamic-draft', action='store_true')
    parser.add_argument("--use-tile-constraint", type=str, default="none",
                        choices=["none", "cut-128"])
    parser.add_argument('--use-lazy-draft-kv-cache', action='store_true')
    parser.add_argument('--predictor-degree', type=int, default=3)
    parser.add_argument('--predictor-agg-type', type=str, default='median')
    parser.add_argument('--use-target-attention', action='store_true')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument(
        '--quantization', choices=['awq', 'squeezellm', None], default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--temperature', type=float,
                        default=0.75, help='Sampling temperature.')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16',
                 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model

    main(args)
