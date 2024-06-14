from typing import List, Optional, Union
import torch.multiprocessing as mp
from typing import Any
import asyncio
from functools import partial

from vllm.engine.arg_utils import SpecDecodeEngineArgs
from vllm.engine.spec_decode_llm_engine import SpecDecodeLLMEngine
from vllm.utils import get_ip, get_open_port, nvtx_range
from vllm.sampling_params import SamplingParams

# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)


class AysncSpecDecodeLLMEngine:
    def __init__(self,
                 target_model: str,
                 draft_model: str,
                 draft_size: int,
                 collocate: bool = False,
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
                 ** kwargs):

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        engine_args = SpecDecodeEngineArgs(
            target_model=target_model,
            draft_model=draft_model,
            draft_size=draft_size,
            collocate=collocate,
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
            ** kwargs,
        )

        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=init_engine,
                             args=(child_conn, engine_args))
        process.start()

        self.llm_engine_process = process
        self.llm_engine_pipe = parent_conn

    def run_step_async(self) -> None:
        self.llm_engine_pipe.send(("step", [], {}))

    def get_step_async_output(self) -> Any:
        return self.llm_engine_pipe.recv()

    def run_add_request_async(self,
                              request_id: str,
                              prompt: Optional[str],
                              sampling_params: SamplingParams,
                              prompt_token_ids: Optional[List[int]] = None,
                              arrival_time: Optional[float] = None) -> Any:

        self.llm_engine_pipe.send(("add_request", [
            request_id, prompt, sampling_params, prompt_token_ids, arrival_time
        ], {}))


def init_engine(pipe,
                engine_args: SpecDecodeEngineArgs):
    llm_engine = SpecDecodeLLMEngine.from_engine_args(engine_args)

    while True:
        method, args, kwargs = pipe.recv()

        if method == "shutdown":
            break

        result = getattr(llm_engine, method)(*args, **kwargs)
        pipe.send(result)
