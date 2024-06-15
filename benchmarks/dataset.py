import json
import os

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset

MAX_LENGTH = 1000

DATASET_DIR = "/home/noppanat/workspace/datasets"


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

    if len(filtered_dataset) > MAX_LENGTH:
        filtered_dataset = filtered_dataset[:MAX_LENGTH]

    return filtered_dataset


def load_gsm8k(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('gsm8k', 'main')['train']
    return process_dataset(dataset, 'question', 'answer', tokenizer)


def load_humaneval(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('openai_humaneval')['test']
    return process_dataset(dataset, 'prompt', 'canonical_solution', tokenizer)


def load_alpaca(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('tatsu-lab/alpaca')['train']
    return process_dataset(dataset, 'instruction', 'output', tokenizer, input_key='input')


def load_mt_bench(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('philschmid/mt-bench')['train']
    prompts = [data['turns'][0] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    completions = []
    with open(f'/gpt-4.jsonl', 'r') as file:
        for line in file:
            json_object = json.loads(line)
            completions.append(json_object['choices'][0]['turns'][0])
    completion_token_ids = tokenizer(completions).input_ids

    return filter_and_process(prompts, prompt_token_ids, completion_token_ids)


def load_sharegpt(tokenizer: PreTrainedTokenizerBase):
    with open(f'{DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json') as f:
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
    dataset = load_dataset('codeparrot/apps')['train']
    return process_dataset(dataset, 'question', 'solutions', tokenizer)


def load_dialogue(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('facebook/empathetic_dialogues')['train']
    return process_dataset(dataset, 'prompt', 'utterance', tokenizer)


def load_chatbot(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('alespalla/chatbot_instruction_prompts')['train']
    return process_dataset(dataset, 'prompt', 'response', tokenizer)


def load_finance(tokenizer: PreTrainedTokenizerBase):
    dataset = load_dataset('gbharti/finance-alpaca')['train']
    return process_dataset(dataset, 'instruction', 'output', tokenizer)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/opt-6.7b', trust_remote_code=True)

    # Test GSM8K dataset
    print("Loading GSM8K dataset...")
    gsm8k_data = load_gsm8k(tokenizer)
    print("GSM8K Sample Data Length:", len(gsm8k_data))

    # Test HumanEval dataset
    print("Loading HumanEval dataset...")
    humaneval_data = load_humaneval(tokenizer)
    print("HumanEval Sample Data Length:", len(humaneval_data))

    # Test Alpaca dataset
    print("Loading Alpaca dataset...")
    alpaca_data = load_alpaca(tokenizer)
    print("Alpaca Sample Data Length:", len(alpaca_data))

    # Test MT-Bench dataset
    print("Loading MT-Bench dataset...")
    mt_bench_data = load_mt_bench(tokenizer)
    print("MT-Bench Sample Data Length:", len(mt_bench_data))

    # Test ShareGPT dataset
    print("Loading ShareGPT dataset...")
    sharegpt_data = load_sharegpt(tokenizer)
    print("ShareGPT Sample Data Length:", len(sharegpt_data))

    # Test APPS dataset
    print("Loading APPS dataset...")
    apps_data = load_apps(tokenizer)
    print("APPS Sample Data Length:", len(apps_data))

    # Test Dialogue dataset
    print("Loading Dialogue dataset...")
    dialogue_data = load_dialogue(tokenizer)
    print("Dialogue Sample Data Length:", len(dialogue_data))

    # Test Chatbot dataset
    print("Loading Chatbot dataset...")
    chatbot_data = load_chatbot(tokenizer)
    print("Chatbot Sample Data Length:", len(chatbot_data))

    # Test Finance dataset
    print("Loading Finance dataset...")
    finance_data = load_finance(tokenizer)
    print("Finance Sample Data Length:", len(finance_data))


if __name__ == '__main__':
    main()
