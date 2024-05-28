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
from torch.cuda import nvtx
import matplotlib.pyplot as plt

# Change to the desired GPU ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# current date 
# current_time = time.strftime("%Y%m%d")

# folder indicator
folder_indicator = '5_27'

# Get NVIDIA GPU name 
gpu_name = torch.cuda.get_device_name(0)

# Constants
DOWNLOAD_DIR = '/mnt/sda/download'
OUTPUT_DIR = f'/mnt/sda/results/{gpu_name}/{folder_indicator}'
PREDICTOR_PATH = 'predictor_5_28'
MAX_NUM_SEQUENCE = 10000
MAX_NUM_ITERATION = 10

# Test cases
STATIC = True # all static draft sizes
STATIC_0 = False # just static draft size 0
STATIC_TILE = True
DYNAMIC = True
DYNAMIC_TILE_CUT = True
DYNAMIC_TILE_SKIP = True


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
        max_tokens=256,
    )

    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "none"

    llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                 sampling_params=dummy_sampling_params, use_tqdm=False)

def plot_distribution(num_batched_tokens, throughput_mean, idx, args):
    print("Plotting the distribution of num_batched_tokens...")

    result = {}

    for key in num_batched_tokens.keys():
        tokens = num_batched_tokens[key][idx]

        total_tokens = []
        for i in range(len(tokens)):
            total_tokens.append(sum(tokens[i]))

        speedup = throughput_mean[key][idx] / throughput_mean["static_0"][idx]

        result[key] = [total_tokens, tokens, speedup]

    # plot the figure subplots with length of keys
    fig, axs = plt.subplots(1, len(result.keys()), figsize=(50, 3))

    for i, key in enumerate(result.keys()):
        axs[i].hist(result[key][0], bins=range(1, 256, 1), density=True, color='blue')
        axs[i].set_title(f"{key} - Speedup: {result[key][2]:.3f}")
        axs[i].axvline(x=128, color='red', linestyle='--', linewidth=2, label='128')
        axs[i].axvline(x=256, color='red', linestyle='--', linewidth=2, label='256')
        axs[i].set_xticks(np.arange(0, 257, 16))

    cleaned_target_model = args.target_model.split("/")[-1]
    cleaned_draft_model = args.draft_model.split("/")[-1]
    cleaned_temperature = str(args.temperature).replace(".", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    distribution_file_name = f"batch_{idx}_distribution_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}_{args.output_len}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, distribution_file_name))
    plt.close()

    for key in ["dynamic", "dynamic_tile_cut", "dynamic_tile_skip"]:
        data = result[key][1]
        num_lists = len(data)
        fig, axes = plt.subplots(num_lists, 1, figsize=(10, 5 * num_lists))

        for i, (sublist, ax) in enumerate(zip(data, axes)):
            # Calculate the frequency of each number from 1 to 8
            counts = [sublist.count(i) for i in range(1, 9)]

            # Plot the frequency distribution
            ax.bar(range(1, 9), counts, color='blue', edgecolor='black')

            # Add labels and title
            ax.set_xlabel('Number')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{i + 1}')
            ax.set_xticks(range(1, 9))

        file_name = f"{key}_batch_{idx}_distribution_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}_{args.output_len}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, file_name))
        plt.close()
    
def save_results_to_csv(throughputs_mean, throughputs_std, args):
    print("Saving results to CSV...")
    # Calculate the speedup compared to static_0 
    speedup = {}
    for key in throughputs_mean.keys():
        speedup[key] = [throughputs_mean[key][i] / throughputs_mean["static_0"][i] for i in range(len(throughputs_mean[key]))]

    # print the results with pprint
    # pprint(speedup)

    # save speedup to csv file 
    keys = list(throughputs_mean.keys())
    header = ", ".join(keys)

    max_length = max(len(value) for value in throughputs_mean.values())
    rows = [
        ", ".join(f"{speedup[key][i]:.3f}" if i < len(
            speedup[key]) else "" for key in keys)
        for i in range(max_length)
    ]

    csv_output = header + "\n" + "\n".join(rows)

    cleaned_target_model = args.target_model.split("/")[-1] 
    cleaned_draft_model = args.draft_model.split("/")[-1]
    cleaned_temperature = str(args.temperature).replace(".", "_")

    os.makedirs(OUTPUT_DIR, exist_ok=True)  

    speedup_result_file_name = f"speedup_results_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}_{args.output_len}.csv"

    with open(os.path.join(OUTPUT_DIR, speedup_result_file_name), "w") as f:
        f.write(csv_output)

    total_result_file_name = f"total_results_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}_{args.batch_size}_{args.output_len}.csv"

    rows = [
        ", ".join(f"{throughputs_mean[key][i]:.3f}" if i < len(
            throughputs_mean[key]) else "" for key in keys)
        for i in range(max_length)
    ]

    csv_output = header + "\n" + "\n".join(rows)

    with open(os.path.join(OUTPUT_DIR, total_result_file_name), "w") as f:
        f.write(csv_output)


def benchmark(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    args.use_lookup_table = True

    llm = SpSLLM(
        target_model=args.target_model,
        draft_model=args.draft_model,
        draft_size=args.draft_size,
        use_dynamic_draft_size=args.use_dynamic_draft,
        use_tile_constraint=args.use_tile_constraint,
        use_lazy_draft_kv_cache=True,
        predictor_degree=args.predictor_degree,
        predictor_agg_type=args.predictor_agg_type,
        use_lookup_table=args.use_lookup_table,
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

    if args.use_lookup_table:
        predictor_path = f"{PREDICTOR_PATH}/lookup_{args.predictor_agg_type}_{cleaned_target_model}_{cleaned_draft_model}_{args.dataset}"
    else:
        predictor_path = f"{PREDICTOR_PATH}/predictor_degree{args.predictor_degree}_{args.predictor_agg_type}_{cleaned_target_model}_{cleaned_draft_model}_{cleaned_temperature}_{args.dataset}.csv"
    # predictor_path = f"{PREDICTOR_PATH}/predictor_degree2_median_opt-6.7b_opt-125m_0_0_apps.csv"
    llm.llm_engine.workers[0].draft_optimizer.initialize(predictor_path)

    if llm.llm_engine.workers[0].draft_optimizer.retrain:
        for temperature in [0, 0.25, 0.5, 0.75, 1.0]:
            pretrain(llm, requests, args, temperature)
        llm.llm_engine.workers[0].draft_optimizer.save_predictor()

    import sys
    sys.exit()

    throughputs_mean = {f"static_{i}": [] for i in range(8)}
    throughputs_mean.update({"static_tile": [], "dynamic": [], "dynamic_tile_cut": [], "dynamic_tile_skip": []})

    throughputs_std = {f"static_{i}": [] for i in range(8)}
    throughputs_std.update({"static_tile": [], "dynamic": [], "dynamic_tile_cut": [], "dynamic_tile_skip": []})

    num_batched_tokens = {f"static_{i}": [] for i in range(8)}
    num_batched_tokens.update({"static_tile": [], "dynamic": [], "dynamic_tile_cut": [], "dynamic_tile_skip": []})

    # Adjust the requests to be multiples of the batch size
    adjusted_max_num_sequence = (
        min(MAX_NUM_SEQUENCE, len(requests)) // args.batch_size) * args.batch_size

    if adjusted_max_num_sequence > args.batch_size * MAX_NUM_ITERATION:
        adjusted_max_num_sequence = args.batch_size * MAX_NUM_ITERATION

    idx = 0
    for i in tqdm(range(0, adjusted_max_num_sequence, args.batch_size)):
        sampled_requests = requests[i:i + args.batch_size]

        if DYNAMIC_TILE_CUT:
            nvtx.range_push("dynamic_tile_cut")
            benchmark_dynamic_tile_cut(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()

        if DYNAMIC:
            nvtx.range_push("dynamic")
            benchmark_dynamic_draft(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()

        if STATIC:
            nvtx.range_push("static_draft")
            benchmark_static_draft(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()
        
        if STATIC_0:
            nvtx.range_push("static_draft_0")
            benchmark_static_draft_0(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()

        if STATIC_TILE:
            nvtx.range_push("static_tile")
            benchmark_static_tile(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()

        if DYNAMIC_TILE_SKIP:
            nvtx.range_push("dynamic_tile_skip")
            benchmark_dynamic_tile_skip(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args)
            nvtx.range_pop()

        # plot the distribution of num_batched_tokens
        # plot_distribution(num_batched_tokens, throughputs_mean, idx, args)
        # idx += 1

    save_results_to_csv(throughputs_mean, throughputs_std, args)

def pretrain(llm, requests, args, temperature):
    print("Pretraining the predictor... for temperature: ", temperature)
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "none"
    llm.llm_engine.scheduler.num_batched_tokens = []

    # Adjust the requests to be multiples of the batch size
    save_batch_size = args.batch_size
    args.batch_size = 32
    adjusted_max_num_sequence = (
        min(MAX_NUM_SEQUENCE, len(requests)) // args.batch_size) * args.batch_size
    
    for i in tqdm(range(0, adjusted_max_num_sequence, args.batch_size)):
        sampled_requests = requests[i:i + args.batch_size]
        for prompt, _, output_len in sampled_requests:
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=512,  # currently fixed to 512 but can be changed to output_len
            )
            llm._add_request(prompt=prompt, prompt_token_ids=None,
                             sampling_params=sampling_params)

        llm._run_engine(use_tqdm=False)

    args.batch_size = save_batch_size

def benchmark_static_draft(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "none"

    for draft_size in range(0, 8):
        llm.llm_engine.sps_config.draft_size = draft_size
        llm.llm_engine.scheduler.num_batched_tokens = []

        print(f"static_{draft_size}")
        benchmark_requests(llm, sampled_requests,
                           throughputs_mean[f"static_{draft_size}"], throughputs_std[f"static_{draft_size}"], num_batched_tokens[f"static_{draft_size}"], args)

def benchmark_static_draft_0(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "none"

    for draft_size in range(0, 1):
        llm.llm_engine.sps_config.draft_size = draft_size
        llm.llm_engine.scheduler.num_batched_tokens = []

        print("static_0")
        benchmark_requests(llm, sampled_requests,
                           throughputs_mean[f"static_{draft_size}"], throughputs_std[f"static_{draft_size}"], num_batched_tokens[f"static_{draft_size}"], args)


def benchmark_static_tile(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = False
    llm.llm_engine.sps_config.use_tile_constraint = "cut-128"
    llm.llm_engine.sps_config.draft_size = 7
    llm.llm_engine.scheduler.num_batched_tokens = []

    print("static_tile")
    benchmark_requests(llm, sampled_requests, throughputs_mean["static_tile"], throughputs_std["static_tile"], num_batched_tokens["static_tile"], args)

def benchmark_dynamic_draft(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "none"
    llm.llm_engine.scheduler.num_batched_tokens = []

    print("dynamic")
    benchmark_requests(llm, sampled_requests, throughputs_mean["dynamic"], throughputs_std["dynamic"], num_batched_tokens["dynamic"], args)

def benchmark_dynamic_tile_cut(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "cut-128"
    llm.llm_engine.scheduler.num_batched_tokens = []

    print("dynamic_tile_cut")
    benchmark_requests(llm, sampled_requests,
                       throughputs_mean["dynamic_tile_cut"], throughputs_std["dynamic_tile_cut"], num_batched_tokens["dynamic_tile_cut"], args)

def benchmark_dynamic_tile_skip(llm, sampled_requests, throughputs_mean, throughputs_std, num_batched_tokens, args):
    llm.llm_engine.sps_config.use_dynamic_draft_size = True
    llm.llm_engine.sps_config.use_tile_constraint = "skip-128-192"
    llm.llm_engine.scheduler.num_batched_tokens = []

    print("dynamic_tile_skip")
    benchmark_requests(llm, sampled_requests,
                       throughputs_mean["dynamic_tile_skip"], throughputs_std["dynamic_tile_skip"], num_batched_tokens["dynamic_tile_skip"], args)
    

def benchmark_requests(llm: SpSLLM, sampled_requests, throughput_mean_list, throughput_std_list, num_batched_tokens_list, args):
    # Run 5 times and take the average throughput
    result = []
    for _ in range(1):
        for prompt, _, output_len in sampled_requests:
            sampling_params = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=1.0,
                use_beam_search=False,
                ignore_eos=True,
                max_tokens=args.output_len,  # currently fixed to 512 but can be changed to output_len
            )
            llm._add_request(prompt=prompt, prompt_token_ids=None,
                            sampling_params=sampling_params)

        torch.cuda.synchronize()
        start_time = time.monotonic()
        outputs = llm._run_engine(use_tqdm=False)
        torch.cuda.synchronize()
        end_time = time.monotonic()

        # Print output text
        # print("--" * 50, len(outputs[0].outputs[0].token_ids))
        # print(outputs[0].outputs[0].text)
        # print("*" * 40)
        # print(outputs[0].outputs[0].token_ids)

        total_tokens = sum(len(output.outputs[0].token_ids) +
                        sampled_requests[idx][1] for idx, output in enumerate(outputs))
        result.append(total_tokens / (end_time - start_time))
        num_batched_tokens_list.append(llm.llm_engine.scheduler.num_batched_tokens)
        llm.llm_engine.scheduler.num_batched_tokens = []
                              
    # print std_dev
    print(np.mean(result), np.std(result), result)
    throughput_mean_list.append(np.mean(result))
    throughput_std_list.append(np.std(result))


def main(args: argparse.Namespace):
    benchmark(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of requests till completion.')
    parser.add_argument("--dataset", type=str, default="alpaca",
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
    parser.add_argument('--use-lookup-table', action='store_true')
    parser.add_argument('--use-target-attention', action='store_true')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument(
        '--quantization', choices=['awq', 'squeezellm', None], default=None)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--draft-size', type=int, default=7)
    parser.add_argument('--temperature', type=float,
                        default=0.75, help='Sampling temperature.')
    parser.add_argument('--output-len', type=int, default=512)
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
