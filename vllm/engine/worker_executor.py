import copy
import torch.multiprocessing as mp
from typing import Any

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
        )

        self.draft_worker.init_model()
        self.draft_worker.load_model()

        parent_conn, child_conn = mp.Pipe()

        process = mp.Process(target=init_worker, args=(child_conn, target_model_config,
                                                       parallel_config, scheduler_config, spec_decode_config))
        process.start()

        self.target_worker_process = process
        self.target_worker_pipe = parent_conn

    @ nvtx_range("run_draft_worker_sync")
    def run_draft_worker_sync(self, method: str, *args, **kwargs) -> Any:
        worker_instance = self.draft_worker
        return getattr(worker_instance, method)(*args, **kwargs)

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

    def shutdown(self) -> None:
        self.target_worker_pipe.send(("shutdown", [], {}))
        self.target_worker_process.join()


def init_worker(pipe, target_model_config, parallel_config, scheduler_config, spec_decode_config):
    worker_instance = SpecDecodeWorker(
        copy.deepcopy(target_model_config),
        copy.deepcopy(parallel_config),
        copy.deepcopy(scheduler_config),
        copy.deepcopy(spec_decode_config),
        local_rank=0,
        rank=0,
        distributed_init_method=f"tcp://{get_ip()}:{get_open_port()}",
    )
    worker_instance.init_model()
    worker_instance.load_model()

    while True:
        method, args, kwargs = pipe.recv()

        if method == "shutdown":
            break

        result = getattr(worker_instance, method)(*args, **kwargs)
        pipe.send(result)
