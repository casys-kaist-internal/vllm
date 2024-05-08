"""Print length of the datsets."""

import argparse
import json

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset


def load_gsm8k(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset("gsm8k", "main")["train"]

    # Tokenize the prompts and completions.
    prompts = [data["question"] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data["answer"] for data in dataset]
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
    dataset = load_dataset("openai_humaneval")["test"]

    # Tokenize the prompts and completions.
    prompts = [data["prompt"] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data["canonical_solution"] for data in dataset]
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
    dataset = load_dataset("tatsu-lab/alpaca")["train"]

    # Tokenize the prompts and completions.
    prompts = [data["instruction"] + data["input"] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data["output"] for data in dataset]
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
    dataset = load_dataset("philschmid/mt-bench")["train"]
    prompts = [data["turns"][0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    completions = []
    # Open the output of gpt-4 mt-bench file to get completion token ids
    with open("gpt-4.jsonl", "r") as file:
        # Iterate over each line in the file
        for line in file:
            # Parse the JSON data from each line
            json_object = json.loads(line)
            completions.append(json_object["choices"][0]["turns"][0])
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
    with open(
        "/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    ) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
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
    dataset = load_dataset("codeparrot/apps")["train"]

    # Tokenize the prompts and completions.
    prompts = [data["question"] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data["solutions"] for data in dataset]
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


def main(args: argparse.Namespace):
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    print("gsm8k:", len(load_gsm8k(tokenizer)))
    print("humaneval:", len(load_humaneval(tokenizer)))
    print("alpaca:", len(load_alpaca(tokenizer)))
    print("mt-bech:", len(load_mt_bench(tokenizer)))
    print("sharegpt:", len(load_sharegpt(tokenizer)))
    print("apps:", len(load_apps(tokenizer)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the lengths of the datasets")
    parser.add_argument("--tokenizer", type=str, default="facebook/opt-6.7b")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
