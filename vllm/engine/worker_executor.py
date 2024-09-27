import copy
import torch.multiprocessing as mp
import torch
from typing import Any
import gc

from vllm.worker.spec_decode_worker import SpecDecodeWorker
from vllm.utils import get_ip, get_open_port, nvtx_range

# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)


class WorkerExecutor:
    def __init__(self,
                 target_model_config,
                 draft_model_config,
                 parallel_config,
                 scheduler_config,
                 spec_decode_config):
        # If draft_size is 0, we only need to initialize the target worker.
        # We initialize the target worker in the main process.
        # If draft_size is not 0, we need to initialize both the draft and target workers.
        # We initialize the target worker in a separate process and
        # draft worker in the main process.
        self.only_target = (spec_decode_config.draft_size == 0)
        # self.only_target = False

        if self.only_target:
            self.target_worker = SpecDecodeWorker(
                copy.deepcopy(target_model_config),
                copy.deepcopy(parallel_config),
                copy.deepcopy(scheduler_config),
                copy.deepcopy(spec_decode_config),
                local_rank=0,
                rank=0,
                distributed_init_method=f"tcp://{get_ip()}:{get_open_port()}",
                is_target=True
            )
            self.target_worker.init_model()
            self.target_worker.load_model()

        else:
            self.task_queue = mp.SimpleQueue()
            self.result_queue = mp.SimpleQueue()

            process = mp.Process(target=init_worker, args=(self.task_queue, self.result_queue, target_model_config,
                                                           parallel_config, scheduler_config, spec_decode_config))
            process.start()

            self.target_worker_process = process

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

    # For sync engine
    @nvtx_range("run_draft_worker_sync")
    def run_draft_worker_sync(self, method: str, *args, **kwargs) -> Any:
        assert self.only_target is False
        return getattr(self.draft_worker, method)(*args, **kwargs)

    @nvtx_range("run_target_worker_sync")
    def run_target_worker_sync(self, method: str, *args, **kwargs) -> Any:
        if self.only_target:
            return getattr(self.target_worker, method)(*args, **kwargs)
        else:
            self.task_queue.put((method, args, kwargs))
            # gc.get_objects()
            result = self.result_queue.get()
            return result

    def run_target_worker_async(self, method: str, *args, **kwargs) -> None:
        """
        Send task to target worker asynchronously.
        """
        assert self.only_target is False
        self.task_queue.put((method, args, kwargs))

    def check_target_worker_async_done(self) -> bool:
        """
        Check if the target worker has finished the task.
        """
        assert self.only_target is False
        return not self.result_queue.empty()

    def get_target_worker_async_output(self) -> Any:
        """
        Receive the output from the target worker.
        """
        assert self.only_target is False
        # gc.get_objects()
        return self.result_queue.get()

    def shutdown(self) -> None:
        if not self.only_target:
            self.task_queue.put(("shutdown", [], {}))
            self.target_worker_process.join()


def init_worker(task_queue, result_queue, target_model_config, parallel_config, scheduler_config, spec_decode_config):
    # Run target worker in different stream
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
        method, args, kwargs = task_queue.get()

        if method == "shutdown":
            break

        result = getattr(target_worker, method)(*args, **kwargs)
        result_queue.put(result)
