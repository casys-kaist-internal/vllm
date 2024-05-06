from typing import List, Optional, Union

import torch
from torch.cuda import nvtx
import time

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import SpSEngineArgs
from vllm.engine.sps_llm_engine import SpSLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.sequence import SpSStage


class SpSLLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
    """

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        draft_size: int,
        tile_size: int,
        use_dynamic_draft_size: bool = False,
        use_tile_size_constraint: bool = False,
        use_lazy_draft_kv_cache: bool = True,
        use_target_attention: bool = False,
        target_draft_latency_ratio: float = 0.2,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.95,
        max_num_batched_tokens: int = None,
        swap_space: int = 4,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = SpSEngineArgs(
            target_model=target_model,
            draft_model=draft_model,
            draft_size=draft_size,
            tile_size=tile_size,
            use_dynamic_draft_size=use_dynamic_draft_size,
            use_tile_size_constraint=use_tile_size_constraint,
            use_target_attention=use_target_attention,
            use_lazy_draft_kv_cache=use_lazy_draft_kv_cache,
            target_draft_latency_ratio=target_draft_latency_ratio,
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
            max_num_batched_tokens=max_num_batched_tokens,
            swap_space=swap_space,
            **kwargs,
        )
        self.llm_engine = SpSLLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs, sps_stage = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    # Return at first output
                    self.llm_engine.abort_all_requests()
                    return step_outputs
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
    
    def _run_profile(self) -> None:
        batch_size_list = range(1, 10)
        # Initialize tqdm.
        num_requests = len(batch_size_list)
        pbar = tqdm(total=num_requests, desc="Profiling latencies")

        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=128,
        )

        dummy_prompt_token_ids = [0] * 32

        # for batch_size in range(1, self.llm_engine.scheduler_config.max_num_seqs + 1):
        for batch_size in range(1, 128):
            for _ in range(batch_size):
                self._add_request(None, sampling_params, dummy_prompt_token_ids)

            sps_stage = None
            # Warmup
            while sps_stage is not SpSStage.TARGET_DECODE:
                _, sps_stage = self.llm_engine.step()
        
            draft_latencies = []
            target_latencies = []

            iteration = 10
            for _ in range(iteration):
                torch.cuda.synchronize()
                draft_start = time.monotonic()
                _, sps_stage = self.llm_engine.step()
                torch.cuda.synchronize()
                draft_end = time.monotonic()
                draft_latencies.append(draft_end - draft_start)
                assert sps_stage is SpSStage.DRAFT_DECODE

                torch.cuda.synchronize()
                target_start = time.monotonic()
                _, sps_stage = self.llm_engine.step()
                torch.cuda.synchronize()
                target_end = time.monotonic()
                target_latencies.append(target_end - target_start)
                assert sps_stage is SpSStage.TARGET_DECODE

            self.llm_engine.abort_all_requests()

            self.llm_engine.draft_latencies[batch_size] = (sum(draft_latencies) / len(draft_latencies)) / self.llm_engine.sps_config.draft_size
            self.llm_engine.target_latencies[batch_size] = sum(target_latencies) / len(target_latencies)
            pbar.update(1)

            print("-"*80)
            print(f"Batch size: {batch_size}")
            print(f"Draft latency: {self.llm_engine.draft_latencies[batch_size]}")
            print(f"Target latency: {self.llm_engine.target_latencies[batch_size]}")

        print(self.llm_engine.draft_latencies)
        print(self.llm_engine.target_latencies)


    def _run_engine_benchmark(self) -> List[RequestOutput]:
        start = None
        done = False
        while self.llm_engine.has_unfinished_requests() and not done:
            step_outputs, sps_stage = self.llm_engine.step()
            if sps_stage is not SpSStage.PROMPT and start is None:
                torch.cuda.synchronize()
                start = time.monotonic()

            num_free_gpu_blocks = self.llm_engine.scheduler.block_manager.gpu_allocator.get_num_free_blocks()
            if num_free_gpu_blocks < len(step_outputs):
                torch.cuda.synchronize()
                end = time.monotonic()
                self.llm_engine.abort_all_requests()
                done = True
            else:
                for output in step_outputs:
                    # return when one request is finished
                    # since we should be exact about batch size
                    if output.finished:
                        torch.cuda.synchronize()
                        end = time.monotonic()
                        self.llm_engine.abort_all_requests()
                        done = True
                        print(output.outputs[0].text)

        output_len = []
        for output in step_outputs:
            # remove the first and second output token which is
            # generated by prompt phase and the first generation phase
            output_len.append(len(output.outputs[0].token_ids) - 2)

        return end - start, output_len
