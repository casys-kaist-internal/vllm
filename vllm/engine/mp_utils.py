import copy

from vllm.worker.spec_decode_worker import SpecDecodeWorker
from vllm.utils import get_ip, get_open_port, nvtx_range


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
