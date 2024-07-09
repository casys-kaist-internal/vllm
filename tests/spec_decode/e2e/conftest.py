import asyncio
import time
from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import ray
import torch

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
DOWNLOAD_DIR = '/home/noppanat/workspace/models'


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
        "The future of AI is",
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
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

    spec_batch_tokens, spec_batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    (baseline_batch_tokens,
     baseline_batch_token_ids) = get_output_from_llm_generator(
         baseline_llm_generator, prompts, sampling_params)

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


# ----------- Below are helper functions for testing serving mode -----------#

def get_requests_with_time(input_requests: List[str],
                           request_rate: float) -> List[Tuple[float, str]]:
    input_requests = iter(input_requests)
    requests_with_time = []
    current_time = 0.0

    for request in input_requests:
        requests_with_time.append((current_time, request))

        if request_rate == float("inf"):
            # If the request rate is infinity, then all requests are sent at time 0.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # Accumulate the interval to get the next request time.
        current_time += interval

    return requests_with_time


def get_serving_output_from_llm_generator(llm_generator,
                                          prompts,
                                          request_rate: float,
                                          get_sampling_params: Callable[[], SamplingParams],
                                          ) -> Tuple[List[str], List[List[int]]]:
    tokens: List[str] = []
    token_ids: List[List[int]] = []
    llm: SpecDecodeLLM
    for llm in llm_generator():
        requests_with_time = get_requests_with_time(prompts, request_rate)
        # Ensure the list is sorted by time
        requests_with_time.sort(reverse=True)

        start_time = time.perf_counter()

        while requests_with_time:
            current_time = time.perf_counter() - start_time

            # Add requests to the engine if their scheduled time has passed
            while requests_with_time and (requests_with_time[-1][0] <= current_time):
                _, prompt = requests_with_time.pop()
                sampling_params = get_sampling_params()
                llm._add_request(prompt=prompt,
                                 prompt_token_ids=None,
                                 sampling_params=sampling_params)

            step_outputs = llm.llm_engine.step()
            token_ids.extend(
                output.outputs[0].token_ids for output in step_outputs)
            tokens.extend(output.outputs[0].text for output in step_outputs)

        del llm

    return tokens, token_ids


def run_greedy_serving_correctness_test(baseline_llm_generator,
                                        test_llm_generator,
                                        num_requests: int,
                                        max_output_len: int,
                                        request_rate: float,
                                        force_output_len: bool,
                                        print_tokens: bool = True):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM in the serving mode. It asserts greedy equality, e.g. that the 
    outputs are exactly the same when temperature is zero.
    """
    def get_sampling_params():
        # If the test requires that we generated max_output_len tokens, then set the
        # sampling params to ignore eos token.
        ignore_eos = force_output_len
        temperature = 0.0
        return SamplingParams(
            n=1,
            temperature=temperature,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=ignore_eos,
            max_tokens=max_output_len,
        )

    prompts = [
        "The future of AI is",
        "The president of the United States is",
        "Hello, my name is",
        "The capital of France is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(
        cycle(prompts), range(num_requests))]

    (baseline_tokens,
     baseline_token_ids) = get_output_from_llm_generator(
         baseline_llm_generator, prompts, get_sampling_params())

    (spec_tokens,
     spec_token_ids) = get_serving_output_from_llm_generator(
         test_llm_generator, prompts, request_rate, get_sampling_params)

    assert len(baseline_token_ids) == len(prompts)
    assert len(spec_token_ids) == len(prompts)

    for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
            spec_tokens) in enumerate(
                zip(baseline_token_ids, baseline_tokens,
                    spec_token_ids, spec_tokens)):
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
