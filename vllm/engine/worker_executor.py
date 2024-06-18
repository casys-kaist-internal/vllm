import copy
import torch.multiprocessing as mp
from typing import Any
import asyncio
from functools import partial
import torch

from vllm.worker.spec_decode_worker import SpecDecodeWorker
from vllm.utils import get_ip, get_open_port, nvtx_range

# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)


class WorkerExecutor:
    def __init__(self, target_model_config, draft_model_config, parallel_config, scheduler_config, spec_decode_config):
        self.draft_worker = SpecDecodeWorker(
            copy.deepcopy(draft_model_config),
            copy.deepcopy(parallel_config),
            copy.deepcopy(scheduler_config),
            copy.deepcopy(spec_decode_config),
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://{get_ip()}:{get_open_port()}",
            is_target=False
        )

        self.draft_worker.init_model()
        self.draft_worker.load_model()

        parent_conn, child_conn = mp.Pipe()

        process = mp.Process(target=init_worker, args=(child_conn, target_model_config,
                                                       parallel_config, scheduler_config, spec_decode_config))
        process.start()

        self.target_worker_process = process
        self.target_worker_pipe = parent_conn

    # For sync engine
    @ nvtx_range("run_draft_worker_sync")
    def run_draft_worker_sync(self, method: str, *args, **kwargs) -> Any:
        return getattr(self.draft_worker, method)(*args, **kwargs)

    @ nvtx_range("run_target_worker_sync")
    def run_target_worker_sync(self, method: str, *args, **kwargs) -> Any:
        self.target_worker_pipe.send((method, args, kwargs))
        result = self.target_worker_pipe.recv()
        return result

    def run_target_worker_async(self, method: str, *args, **kwargs) -> None:
        """
        Send task to target worker asynchronously.
        """
        self.target_worker_pipe.send((method, args, kwargs))

    def get_target_worker_async_output(self) -> Any:
        """
        Receive the output from the target worker.
        """
        return self.target_worker_pipe.recv()

    # For async engine
    async def _run_draft_worker_sync(self, method: str, *args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        driver_executor = getattr(self.draft_worker, method)
        result = await loop.run_in_executor(None, partial(driver_executor, *args, **kwargs))
        return result

    # Wait the blocking recv in separate thread
    async def _run_target_worker_sync(self, method: str, *args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        self.target_worker_pipe.send((method, args, kwargs))
        result = await loop.run_in_executor(None, self.target_worker_pipe.recv)
        return result

    async def _get_target_worker_async_output(self) -> None:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.target_worker_pipe.recv)
        return result

    def shutdown(self) -> None:
        self.target_worker_pipe.send(("shutdown", [], {}))
        self.target_worker_process.join()


def init_worker(pipe, target_model_config, parallel_config, scheduler_config, spec_decode_config):
    target_stream = torch.cuda.Stream()

    with torch.cuda.stream(target_stream):
        target_worker = SpecDecodeWorker(
            copy.deepcopy(target_model_config),
            copy.deepcopy(parallel_config),
            copy.deepcopy(scheduler_config),
            copy.deepcopy(spec_decode_config),
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://{get_ip()}:{get_open_port()}",
            is_target=True
        )
        target_worker.init_model()
        target_worker.load_model()

        while True:
            method, args, kwargs = pipe.recv()

            if method == "shutdown":
                break

            result = getattr(target_worker, method)(*args, **kwargs)
            pipe.send(result)
