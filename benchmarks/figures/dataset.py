import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)
import random
import json

tokenizer = AutoTokenizer.from_pretrained(
    'facebook/opt-6.7b', trust_remote_code=True)


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
    random.shuffle(filtered_dataset)

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
    random.shuffle(filtered_dataset)

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
    random.shuffle(filtered_dataset)

    return filtered_dataset


def load_sharegpt(tokenizer: PreTrainedTokenizerBase):
    with open('/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
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
    random.shuffle(filtered_dataset)

    return filtered_dataset


# print distribution of input and output length of dataset
gsm8k = load_gsm8k(tokenizer)
humaneval = load_humaneval(tokenizer)
alpaca = load_alpaca(tokenizer)
mt_bench = load_mt_bench(tokenizer)
sharegpt = load_sharegpt(tokenizer)


# Calculate input and output lengths for all datasets
gsm8k_input_len = [data[1] for data in gsm8k]
gsm8k_output_len = [data[2] for data in gsm8k]

humaneval_input_len = [data[1] for data in humaneval]
humaneval_output_len = [data[2] for data in humaneval]

alpaca_input_len = [data[1] for data in alpaca]
alpaca_output_len = [data[2] for data in alpaca]

mt_bench_input_len = [data[1] for data in mt_bench]
mt_bench_output_len = [data[2] for data in mt_bench]

sharegpt_input_len = [data[1] for data in sharegpt]
sharegpt_output_len = [data[2] for data in sharegpt]

mean_values = {
    'alpaca': {'input': np.mean(alpaca_input_len), 'output': np.mean(alpaca_output_len)},
    'gsm8k': {'input': np.mean(gsm8k_input_len), 'output': np.mean(gsm8k_output_len)},
    'humaneval': {'input': np.mean(humaneval_input_len), 'output': np.mean(humaneval_output_len)},
    'mt-bench': {'input': np.mean(mt_bench_input_len), 'output': np.mean(mt_bench_output_len)},
    'sharegpt': {'input': np.mean(sharegpt_input_len), 'output': np.mean(sharegpt_output_len)}
}

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
color = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

# Overlayed Input Length Distribution
for i, input_len in enumerate([gsm8k_input_len, humaneval_input_len, alpaca_input_len, mt_bench_input_len, sharegpt_input_len]):
    dataset_name = ['alpaca', 'gsm8k', 'humaneval', 'mt-bench', 'sharegpt'][i]
    mean_value = mean_values[dataset_name]['input']
    label = f"{dataset_name} (mean={mean_value:.2f})"
    axs[0].hist(input_len, bins=20, color=color[i],
                alpha=0.5, density=True, label=label)
axs[0].set_title('Prompt Length Distribution', fontsize=14)
axs[0].set_xlabel('Length', fontsize=12)
axs[0].set_ylabel('Density', fontsize=12)
axs[0].legend()

# Overlayed Output Length Distribution
for i, output_len in enumerate([gsm8k_output_len, humaneval_output_len, alpaca_output_len, mt_bench_output_len, sharegpt_output_len]):
    dataset_name = ['alpaca', 'gsm8k', 'humaneval', 'mt-bench', 'sharegpt'][i]
    mean_value = mean_values[dataset_name]['output']
    label = f"{dataset_name} (mean={mean_value:.2f})"
    axs[1].hist(output_len, bins=20, color=color[i],
                alpha=0.5, density=True, label=label)
axs[1].set_title('Output Length Distribution', fontsize=14)
axs[1].set_xlabel('Length', fontsize=12)
axs[1].set_ylabel('Density', fontsize=12)
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
plt.savefig('dataset_length_distribution.png')
