import json
import os
from typing import List, Optional, Tuple
import random

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset

DOWNLOAD_DIR = '/home/noppanat/workspace/models'
DATASET_DIR = '/home/noppanat/workspace/datasets'
RANDOM_SAMPLE = False

# Ensure DOWNLOAD_DIR exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def process_dataset(dataset, prompt_key, completion_key, num_prompts, tokenizer, fixed_output_len, input_key=None):
    prompts = [data[prompt_key] +
               (data[input_key] if input_key else '') for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data[completion_key] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    return filter_and_process(prompts, prompt_token_ids, completion_token_ids, num_prompts, fixed_output_len)


def filter_and_process(prompts, prompt_token_ids, completion_token_ids, num_prompts, fixed_output_len):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    tokenized_dataset = []
    for i in range(len(prompts)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    if RANDOM_SAMPLE:
        # Sample the requests.
        filtered_dataset = random.sample(filtered_dataset, num_prompts)
    else:
        if num_prompts > 0 and num_prompts < len(filtered_dataset):
            filtered_dataset = filtered_dataset[:num_prompts]

    return filtered_dataset


def load_gsm8k(num_prompts: int,
               tokenizer: PreTrainedTokenizerBase,
               fixed_output_len: Optional[int] = None):
    dataset = load_dataset('gsm8k', 'main')['train']
    return process_dataset(dataset, 'question', 'answer', num_prompts, tokenizer, fixed_output_len)


def load_humaneval(num_prompts: int,
                   tokenizer: PreTrainedTokenizerBase,
                   fixed_output_len: Optional[int] = None):
    dataset = load_dataset('openai_humaneval')['test']
    return process_dataset(dataset, 'prompt', 'canonical_solution', num_prompts, tokenizer, fixed_output_len)


def load_alpaca(num_prompts: int,
                tokenizer: PreTrainedTokenizerBase,
                fixed_output_len: Optional[int] = None):
    dataset = load_dataset('tatsu-lab/alpaca')['train']
    return process_dataset(dataset, 'instruction', 'output', num_prompts, tokenizer, fixed_output_len, input_key='input')


def load_mt_bench(num_prompts: int,
                  tokenizer: PreTrainedTokenizerBase,
                  fixed_output_len: Optional[int] = None):
    dataset = load_dataset('philschmid/mt-bench')['train']
    prompts = [data['turns'][0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    completions = []
    with open(f'{DATASET_DIR}/gpt-4.jsonl', 'r') as file:
        for line in file:
            json_object = json.loads(line)
            completions.append(json_object['choices'][0]['turns'][0])
    completion_token_ids = tokenizer(completions).input_ids

    return filter_and_process(prompts, prompt_token_ids, completion_token_ids, num_prompts, fixed_output_len)


def load_sharegpt(num_prompts: int,
                  tokenizer: PreTrainedTokenizerBase,
                  fixed_output_len: Optional[int] = None):
    with open(f'{DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
        dataset = json.load(f)

    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    return filter_and_process(prompts, prompt_token_ids, completion_token_ids, num_prompts, fixed_output_len)


def load_apps(num_prompts: int,
              tokenizer: PreTrainedTokenizerBase,
              fixed_output_len: Optional[int] = None):
    dataset = load_dataset('codeparrot/apps')['train']
    return process_dataset(dataset, 'question', 'solutions', num_prompts, tokenizer, fixed_output_len)


def load_dialogue(num_prompts: int,
                  tokenizer: PreTrainedTokenizerBase,
                  fixed_output_len: Optional[int] = None):
    dataset = load_dataset('facebook/empathetic_dialogues')['train']
    return process_dataset(dataset, 'prompt', 'utterance', num_prompts, tokenizer, fixed_output_len)


def load_chatbot(num_prompts: int,
                 tokenizer: PreTrainedTokenizerBase,
                 fixed_output_len: Optional[int] = None):
    dataset = load_dataset('alespalla/chatbot_instruction_prompts')['train']
    return process_dataset(dataset, 'prompt', 'response', num_prompts, tokenizer, fixed_output_len)


def load_finance(num_prompts: int,
                 tokenizer: PreTrainedTokenizerBase,
                 fixed_output_len: Optional[int] = None):
    dataset = load_dataset('gbharti/finance-alpaca')['train']
    return process_dataset(dataset, 'instruction', 'output', num_prompts, tokenizer, fixed_output_len)


def sample_requests(dataset_name: str,
                    num_prompts: int,
                    tokenizer: PreTrainedTokenizerBase,
                    fixed_output_len: Optional[int] = None):
    if dataset_name == 'gsm8k':
        return load_gsm8k(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'humaneval':
        return load_humaneval(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'alpaca':
        return load_alpaca(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'mt-bench':
        return load_mt_bench(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'sharegpt':
        return load_sharegpt(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'apps':
        return load_apps(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'dialogue':
        return load_dialogue(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'chatbot':
        return load_chatbot(num_prompts, tokenizer, fixed_output_len)
    elif dataset_name == 'finance':
        return load_finance(num_prompts, tokenizer, fixed_output_len)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    num_prompts = 50
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/opt-6.7b', trust_remote_code=True)
    fixed_output_len = 128

    # Test GSM8K dataset
    print("Loading GSM8K dataset...")
    gsm8k_data = load_gsm8k(num_prompts, tokenizer, fixed_output_len)
    print("GSM8K Sample Data Length:", len(gsm8k_data))

    # Test HumanEval dataset
    print("Loading HumanEval dataset...")
    humaneval_data = load_humaneval(num_prompts, tokenizer, fixed_output_len)
    print("HumanEval Sample Data Length:", len(humaneval_data))

    # Test Alpaca dataset
    print("Loading Alpaca dataset...")
    alpaca_data = load_alpaca(num_prompts, tokenizer, fixed_output_len)
    print("Alpaca Sample Data Length:", len(alpaca_data))

    # Test MT-Bench dataset
    print("Loading MT-Bench dataset...")
    mt_bench_data = load_mt_bench(num_prompts, tokenizer, fixed_output_len)
    print("MT-Bench Sample Data Length:", len(mt_bench_data))

    # Test ShareGPT dataset
    print("Loading ShareGPT dataset...")
    sharegpt_data = load_sharegpt(num_prompts, tokenizer, fixed_output_len)
    print("ShareGPT Sample Data Length:", len(sharegpt_data))

    # Test APPS dataset
    print("Loading APPS dataset...")
    apps_data = load_apps(num_prompts, tokenizer, fixed_output_len)
    print("APPS Sample Data Length:", len(apps_data))

    # Test Dialogue dataset
    print("Loading Dialogue dataset...")
    dialogue_data = load_dialogue(num_prompts, tokenizer, fixed_output_len)
    print("Dialogue Sample Data Length:", len(dialogue_data))

    # Test Chatbot dataset
    print("Loading Chatbot dataset...")
    chatbot_data = load_chatbot(num_prompts, tokenizer, fixed_output_len)
    print("Chatbot Sample Data Length:", len(chatbot_data))

    # Test Finance dataset
    print("Loading Finance dataset...")
    finance_data = load_finance(num_prompts, tokenizer, fixed_output_len)
    print("Finance Sample Data Length:", len(finance_data))


if __name__ == '__main__':
    main()
