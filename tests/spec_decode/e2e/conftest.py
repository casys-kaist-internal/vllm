import asyncio
import time
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union
import os 

import pytest
import ray
import torch
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from collections import Counter as CT
from scipy.spatial.distance import jensenshannon

from vllm.utils import is_hip

if (not is_hip()):
    from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                        nvmlInit)

from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs, AsyncSpecDecodeEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.async_spec_decode_llm_engine import AsyncSpecDecodeLLMEngine
from vllm.entrypoints.spec_decode_llm import SpecDecodeLLM
from vllm.model_executor.utils import set_random_seed
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter, random_uuid

from ...conftest import cleanup

# DOWNLOAD_DIR = "../models/" # NOTE(noppanat): Change this to the correct path
# NOTE(noppanat): Change this to the correct path
DOWNLOAD_DIR = "/mnt/sda/download"

# Set env variable export CUBLAS_WORKSPACE_CONFIG=:4096:2 export CUBLAS_WORKSPACE_CONFIG=:16:8
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# torch.use_deterministic_algorithms(True)


class AsyncLLM:
    """AsyncLLM

    Note: Current LLM class in vllm don't support async mode, for test purpose,
    we implement async one in here. Maybe we could move to
    vllm/entrypoints/llm.py in future.

    Below AsyncLLM is directly borrow from vllm/entrypoints/llm.py with changes
    to make to work in async mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            # Async engine args
            engine_use_ray=True,
            disable_log_requests=True,
            max_log_len=None,
            **kwargs,
        )
        self.request_counter = Counter()
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:

        if prompts is None:
            raise ValueError("prompts must be provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        if prompts is not None:
            num_requests = len(prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and "
                             "sampling_params must be the same.")

        async def get_output(prompt, sampling_param) -> RequestOutput:
            request_id = random_uuid()
            results_generator = self.llm_engine.generate(
                prompt, sampling_param, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            assert final_output is not None
            return final_output

        outputs: List[RequestOutput] = []
        try:
            for i in range(num_requests):
                prompt = prompts[i] if prompts is not None else None
                res = asyncio.run(get_output(prompt, sampling_params))
                outputs.append(res)
        finally:
            ray.shutdown()
        return outputs


class AsyncSpecDecodeLLM:
    """AsyncSpecDecodeLLM

    Note: Current SpecDecodeLLM class doesn't support async mode, for test purpose,
    we implement async one in here. Maybe we could move to
    vllm/entrypoints/spec_decode_llm.py in future.

    Below AsyncSpecDecodeLLM is directly borrow from vllm/entrypoints/spec_decode_llm.py with changes
    to make to work in async mode.
    """

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        draft_size: int,
        colocate: bool = False,
        enable_chunked_prefill: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        ** kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = AsyncSpecDecodeEngineArgs(
            target_model=target_model,
            draft_model=draft_model,
            draft_size=draft_size,
            colocate=colocate,
            enable_chunked_prefill=enable_chunked_prefill,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            # Async engine args
            engine_use_ray=False,
            disable_log_requests=True,
            max_log_len=None,
            **kwargs,
        )
        self.request_counter = Counter()
        self.llm_engine = AsyncSpecDecodeLLMEngine.from_engine_args(
            engine_args)

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:

        if prompts is None:
            raise ValueError("prompts must be provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        if prompts is not None:
            num_requests = len(prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and "
                             "sampling_params must be the same.")

        async def get_output(prompt, sampling_param) -> RequestOutput:
            request_id = random_uuid()
            results_generator = self.llm_engine.generate(
                prompt, sampling_param, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            assert final_output is not None
            return final_output

        outputs: List[RequestOutput] = []
        try:
            for i in range(num_requests):
                prompt = prompts[i] if prompts is not None else None
                res = asyncio.run(get_output(prompt, sampling_params))
                outputs.append(res)
        finally:
            ray.shutdown()
        return outputs


@pytest.fixture
def baseline_llm_generator(request, common_llm_kwargs, per_test_common_llm_kwargs,
                           baseline_llm_kwargs, test_llm_kwargs,
                           seed):
    return create_llm_generator("baseline", request, common_llm_kwargs,
                                per_test_common_llm_kwargs, baseline_llm_kwargs,
                                test_llm_kwargs, seed)


@pytest.fixture
def test_llm_generator(request, common_llm_kwargs, per_test_common_llm_kwargs,
                       baseline_llm_kwargs, test_llm_kwargs, seed):
    return create_llm_generator("test", request, common_llm_kwargs,
                                per_test_common_llm_kwargs, baseline_llm_kwargs,
                                test_llm_kwargs, seed)


def create_llm_generator(baseline_or_test, request, common_llm_kwargs,
                         per_test_common_llm_kwargs, baseline_llm_kwargs,
                         test_llm_kwargs, seed):
    common_kwargs = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        "download_dir": DOWNLOAD_DIR,
    }
    baseline_kwargs = common_kwargs | baseline_llm_kwargs
    test_kwargs = common_kwargs | test_llm_kwargs
    test_name = request.node.name

    def generator_inner():

        wait_for_gpu_memory_to_clear(
            devices=list(range(torch.cuda.device_count())),
            threshold_bytes=2 * 2**30,
            timeout_s=60,
        )

        use_async = False
        if "use_async" in baseline_kwargs:
            use_async = baseline_kwargs.pop("use_async")
        if "use_async" in test_kwargs:
            use_async = test_kwargs.pop("use_async")
        print(f'{use_async=}')

        print(
            f'Creating {baseline_or_test=} LLM for {test_name=}. {(baseline_kwargs if baseline_or_test == "baseline" else test_kwargs)=}'
        )
        if baseline_or_test == "baseline":
            llm = AsyncLLM(
                **baseline_kwargs) if use_async else LLM(**baseline_kwargs)
        elif baseline_or_test == "test":
            llm = AsyncSpecDecodeLLM(
                **test_kwargs) if use_async else SpecDecodeLLM(**test_kwargs)
        else:
            raise ValueError("Invalid baseline_or_test value.")
        set_random_seed(seed)

        yield llm

        if baseline_or_test == "test":
            llm.llm_engine.worker_executor.shutdown()

        del llm
        cleanup()

    def generator_outer():
        for llm in generator_inner():
            yield llm
            del llm

    return generator_outer


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]]]:
    tokens: List[str] = []
    token_ids: List[List[int]] = []
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]
        del llm

    return tokens, token_ids


def get_logprobs_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> List[List[Dict[int, float]]]:
    """Returns a dict of (token_id: Logprob) for each generated position, for
    each sequence in the batch.
    """
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        logprobs = [output.outputs[0].logprobs[:] for output in outputs]
        del llm

    return logprobs


def run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len: bool,
                                         print_tokens: bool = True):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero.
    """
    temperature = 0

    prompts = [
        "Generate digital startup ideas based on the wish of the people. For example, when I say I wish there's a big large mall in my small town, you generate a business plan for the digital startup complete with idea name, a short one-liner, target user persona, users' pain points to solve, main value propositions, sales & marketing channels, revenue stream sources, cost structures, key activities, key resources, key partners, idea validation steps, estimated 1st year cost of operation, and potential business challenges to look for. Write the result in a markdown table.",
        "The first President of India was Dr. Rajendra Prasad. He served as the President of India from 1950 to 1962, for two consecutive terms. Dr. Prasad was a prominent leader of the Indian Independence Movement and played an important role in shaping India's destiny after independence. He was also a scholar, a jurist, and a Gandhian who believed in the principles of non-violence and truth.",
        "Barack Hussein Obama II was born on August 4, 1961, and is an American politician who served as the 44th president of the United States from 2009 to 2017.",
        "After graduating from high school in 1979, Obama moved to Los Angeles to attend Occidental College on a full scholarship.",
        "UNICEF USA's mission is to relentlessly pursue a more equitable world for every child. UNICEF USA advances the global mission of UNICEF by rallying the American public to support the world's most vulnerable children.",
        "The future of AI is",
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "San Francisco is known for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
        "The best way to learn is to",
        "The most important thing in life is",
        "Alan Turing was a",
        "Elon Musk is the CEO of Tesla and SpaceX. He was born in",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generated max_output_len tokens, then set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
    )

    (baseline_batch_tokens,
     baseline_batch_token_ids) = get_output_from_llm_generator(
         baseline_llm_generator, prompts, sampling_params)
    
    spec_batch_tokens, spec_batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    assert len(baseline_batch_token_ids) == len(prompts)
    assert len(spec_batch_token_ids) == len(prompts)

    for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
            spec_tokens) in enumerate(
                zip(baseline_batch_token_ids, baseline_batch_tokens,
                    spec_batch_token_ids, spec_batch_tokens)):
        if print_tokens:
            print(f'{i=} {baseline_tokens=}')
            print(f'{i=}     {spec_tokens=}')
        print(f'{i=} {baseline_token_ids=}')
        print(f'{i=}     {spec_token_ids=}')

        # compare until the length of baseline_token_ids
        if len(baseline_token_ids) < len(spec_token_ids):
            assert baseline_token_ids == spec_token_ids[:len(
                baseline_token_ids)]
        else:
            assert baseline_token_ids == spec_token_ids


def run_output_distribution_similarity_test(baseline_llm_generator,
                                            test_llm_generator,
                                            temperature,
                                            batch_size,
                                            max_output_len,
                                            force_output_len: bool,
                                            print_tokens: bool = True,
                                            jsd_threshold: float = 0.2,
                                            plot_filename: str = 'token_distribution.png'):
    """Helper method that compares the distributions of output tokens from both
    the baseline LLM and the test LLM using Jensen-Shannon Divergence.
    
    - Matches the output IDs up to the length of the shorter output for each sequence.
    - Draws a PNG file of the two distributions and saves it.
    """

    # List of prompts
    prompts = [
        "Generate digital startup ideas based on the wish of the people. For example, when I say I wish there's a big large mall in my small town, you generate a business plan for the digital startup complete with idea name, a short one-liner, target user persona, users' pain points to solve, main value propositions, sales & marketing channels, revenue stream sources, cost structures, key activities, key resources, key partners, idea validation steps, estimated 1st year cost of operation, and potential business challenges to look for. Write the result in a markdown table.",
        "The first President of India was Dr. Rajendra Prasad. He served as the President of India from 1950 to 1962, for two consecutive terms. Dr. Prasad was a prominent leader of the Indian Independence Movement and played an important role in shaping India's destiny after independence. He was also a scholar, a jurist, and a Gandhian who believed in the principles of non-violence and truth.",
        "Barack Hussein Obama II was born on August 4, 1961, and is an American politician who served as the 44th president of the United States from 2009 to 2017.",
        "After graduating from high school in 1979, Obama moved to Los Angeles to attend Occidental College on a full scholarship.",
        "UNICEF USA's mission is to relentlessly pursue a more equitable world for every child. UNICEF USA advances the global mission of UNICEF by rallying the American public to support the world's most vulnerable children.",
        "The future of AI is",
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "San Francisco is known for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
        "The best way to learn is to",
        "The most important thing in life is",
        "Alan Turing was a",
        "Elon Musk is the CEO of Tesla and SpaceX. He was born in",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generate max_output_len tokens, set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=ignore_eos,
        temperature=temperature,
    )

    # Get outputs from both models
    baseline_batch_tokens, baseline_batch_token_ids = get_output_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params
    )

    spec_batch_tokens, spec_batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params
    )

    # Initialize lists for matched tokens
    baseline_token_ids_matched = []
    spec_token_ids_matched = []

    # Match output IDs up to the length of the shorter output for each sequence
    for seq_baseline, seq_spec in zip(baseline_batch_token_ids, spec_batch_token_ids):
        min_len = min(len(seq_baseline), len(seq_spec))
        baseline_token_ids_matched.extend(seq_baseline[:min_len])
        spec_token_ids_matched.extend(seq_spec[:min_len])

    # Build frequency distributions (histograms) of the token IDs
    baseline_counter = CT(baseline_token_ids_matched)
    spec_counter = CT(spec_token_ids_matched)

    # Get the set of all unique token IDs from both models
    all_tokens = set(baseline_counter.keys()).union(set(spec_counter.keys()))

    # Build frequency arrays
    baseline_freq = np.array([baseline_counter.get(token, 0) for token in all_tokens], dtype=np.float64)
    spec_freq = np.array([spec_counter.get(token, 0) for token in all_tokens], dtype=np.float64)

    # Normalize frequencies to get probability distributions
    baseline_prob = baseline_freq / baseline_freq.sum()
    spec_prob = spec_freq / spec_freq.sum()

    # Calculate Jensen-Shannon Divergence
    js_distance = jensenshannon(baseline_prob, spec_prob)
    js_divergence = js_distance ** 2  # Jensen-Shannon Divergence is the square of the distance

    # Print the JSD value
    print(f"Jensen-Shannon Divergence: {js_divergence}")

    # Determine if the distributions are similar based on the threshold
    if js_divergence > jsd_threshold:
        print("The distributions diverge significantly.")
        test_result = False
    else:
        print("The distributions are similar.")
        test_result = True

    # Plot the distributions and save as a PNG file
    # Get the top N tokens from both counters
    N = 20  # Number of top tokens to display
    baseline_top_tokens = set([token for token, _ in baseline_counter.most_common(N)])
    spec_top_tokens = set([token for token, _ in spec_counter.most_common(N)])
    tokens_for_plot = list(baseline_top_tokens.union(spec_top_tokens))
    tokens_for_plot.sort()  # Sort tokens for consistent ordering

    # Get counts for plotting
    baseline_plot_counts = [baseline_counter.get(token, 0) for token in tokens_for_plot]
    spec_plot_counts = [spec_counter.get(token, 0) for token in tokens_for_plot]

    x = np.arange(len(tokens_for_plot))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, baseline_plot_counts, width, label='Baseline')
    rects2 = ax.bar(x + width / 2, spec_plot_counts, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Token Counts')
    ax.set_title('Token Frequency Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens_for_plot, rotation=90)
    ax.legend()

    # Write the JSD value on the plot
    ax.text(0.5, 0.95, f"Jensen-Shannon Divergence: {js_divergence:.4f}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    
    # Write down pass or fail on the plot
    ax.text(0.5, 0.90, "Pass" if test_result else "Fail",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

    plot_dir = 'dist_plots'

    os.makedirs(plot_dir, exist_ok=True)

    # Concat pass or fail at the front to the plot filename
    plot_filename = f"{'pass' if test_result else 'fail'}_{plot_filename}"

    plot_filename = os.path.join(plot_dir, plot_filename)

    fig.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)

    # Assert the test result
    assert test_result, f"Distributions diverge significantly (JSD={js_divergence})"



def wait_for_gpu_memory_to_clear(devices: List[int],
                                 threshold_bytes: int,
                                 timeout_s: float = 120) -> None:
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    nvmlInit()
    start_time = time.time()
    while True:
        output: Dict[int, str] = {}
        output_raw: Dict[int, float] = {}
        for device in devices:
            dev_handle = nvmlDeviceGetHandleByIndex(device)
            mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
            gb_used = mem_info.used / 2**30
            output_raw[device] = gb_used
            output[device] = f'{gb_used:.02f}'

        print('gpu memory used (GB): ', end='')
        for k, v in output.items():
            print(f'{k}={v}; ', end='')
        print('')

        dur_s = time.time() - start_time
        if all(v <= (threshold_bytes / 2**30) for v in output_raw.values()):
            print(f'Done waiting for free GPU memory on devices {devices=} '
                  f'({threshold_bytes/2**30=}) {dur_s=:.02f}')
            break

        if dur_s >= timeout_s:
            raise ValueError(f'Memory of devices {devices=} not free after '
                             f'{dur_s=:.02f} ({threshold_bytes/2**30=})')

        time.sleep(5)
