"""Benchmark offline inference throughput."""
import numpy as np
import argparse
import random
import contextlib
import time
import torch
import gc
import os
from typing import List, Optional, Tuple
from transformers import (AutoTokenizer)
from dataset import sample_requests
from tabulate import tabulate

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm import LLM, SpecDecodeLLM, SamplingParams
from vllm.utils import is_cpu


DOWNLOAD_DIR = '/mnt/sda/download'

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)


def run(
    requests: List[Tuple[str, int, int]],
    target_model: str,
    draft_model: str,
    draft_size: int,
    colocate: bool,
    chunked_prefill: bool,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
) -> float:

    llm = LLM(
        model=target_model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        enforce_eager=True,
        download_dir=DOWNLOAD_DIR,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    baseline_outputs = llm._run_engine(use_tqdm=True)
    del llm
    cleanup()

    spec_decode_llm = SpecDecodeLLM(
        target_model=target_model,
        draft_model=draft_model,
        draft_size=draft_size,
        colocate=colocate,
        enable_chunked_prefill=chunked_prefill,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        download_dir=DOWNLOAD_DIR,
        disable_bonus_token=True
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        spec_decode_llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    spec_decode_outputs = spec_decode_llm._run_engine(use_tqdm=True)
    spec_decode_llm.llm_engine.worker_executor.shutdown()

    del spec_decode_llm
    cleanup()

    assert len(spec_decode_outputs) == len(baseline_outputs)

    for index, (spec_decode_output, baseline_output) in enumerate(zip(spec_decode_outputs, baseline_outputs)):
        spec_decode_token_ids = spec_decode_output.outputs[0].token_ids
        baseline_token_ids = baseline_output.outputs[0].token_ids
        # print("-" * 80)
        for i in range(len(baseline_token_ids)):
            if spec_decode_token_ids[i] != baseline_token_ids[i]:
                print("TEST FAILED", index)
                print(baseline_token_ids)
                print(spec_decode_token_ids)
                return

    print("TEST PASSED")


def cleanup():
    # TODO(noppanat): Implement vllm/distributed
    destroy_model_parallel()
    # destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
        torch.cuda.empty_cache()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    table = [["target_model", args.target_model],
             ["draft_model", args.draft_model],
             ["draft_size", args.draft_size],
             ["colocate", args.colocate],
             ["chunked_prefill", args.chunked_prefill],
             ["dataset", args.dataset],
             ["input_len", args.input_len],
             ["output_len", args.output_len],
             ["num_prompts", args.num_prompts]]

    print(tabulate(table))

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset == "dummy":
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    run(requests, args.target_model, args.draft_model,
        args.draft_size, args.colocate, args.chunked_prefill,
        args.tokenizer, args.quantization, args.tensor_parallel_size,
        args.seed, args.n, args.use_beam_search,
        args.trust_remote_code, args.dtype,
        args.max_model_len, args.enforce_eager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument('--target-model', type=str,
                        default='facebook/opt-6.7b')
    parser.add_argument('--draft-model', type=str, default='facebook/opt-125m')
    parser.add_argument('--draft-size', type=int, default=4)
    parser.add_argument('--colocate',
                        '-c',
                        action='store_true')
    parser.add_argument('--chunked-prefill',
                        '-cp',
                        action='store_true')
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=32,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.target_model
    if args.dataset is None:
        args.dataset = "dummy"
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    main(args)
