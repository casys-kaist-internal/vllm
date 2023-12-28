"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpSConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import ParallelState
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceGroupMetadata, SpSStage, SequenceData
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.sps_model_runner import SpSModelRunner
from vllm.utils import get_gpu_memory


class SpSWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        target_model_config: ModelConfig,
        draft_model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        sps_config: SpSConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.target_model_config = target_model_config
        self.draft_model_config = draft_model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.sps_config = sps_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        self.target_model_runner = SpSModelRunner(target_model_config, parallel_config,
                                                  scheduler_config)
        self.draft_model_runner = SpSModelRunner(draft_model_config, parallel_config,
                                                 scheduler_config)

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.target_cache_engine = None
        self.draft_cache_engine = None
        self.target_cache_events = None
        self.draft_cache_events = None
        self.target_gpu_cache = None
        self.draft_gpu_cache = None

        self.target_parallel_state = ParallelState()
        self.draft_parallel_state = ParallelState()

    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        _check_if_gpu_supports_dtype(self.target_model_config.dtype)

        # Initialize the distributed environment.
        self._init_distributed_environment()

        # Initialize the model.
        set_random_seed(self.target_model_config.seed)

    def load_model(self):
        self.target_model_runner.load_model(self.target_parallel_state)
        self.draft_model_runner.load_model(self.draft_parallel_state)

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the target model and the draft model.
        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        vocab_size = self.target_model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                sps_stage=SpSStage.PROMPT,
            )
            seqs.append(seq)

        self.target_model_runner.profile_run(seqs)
        self.draft_model_runner.profile_run(seqs)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        target_cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.target_model_config, self.parallel_config)
        draft_cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.draft_model_config, self.parallel_config)  # FIXME(sjchoi)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            (target_cache_block_size + draft_cache_block_size))
        num_cpu_blocks = int(
            cpu_swap_space // (target_cache_block_size + draft_cache_block_size))
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        print("gpu_mem_util", gpu_memory_utilization)
        print("peak_memory", peak_memory)
        print("total_gpu_memory", total_gpu_memory)
        print("target_cache_block_size", target_cache_block_size)
        print("draft_cache_block_size", draft_cache_block_size)
        print("num_gpu_blocks", num_gpu_blocks)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.target_model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.target_cache_engine = CacheEngine(self.cache_config, self.target_model_config,
                                               self.parallel_config)
        self.draft_cache_engine = CacheEngine(self.cache_config, self.draft_model_config,
                                              self.parallel_config)  # FIXME(sjchoi)
        self.target_cache_events = self.target_cache_engine.events
        self.draft_cache_events = self.draft_cache_engine.events
        self.target_gpu_cache = self.target_cache_engine.gpu_cache
        self.draft_gpu_cache = self.draft_cache_engine.gpu_cache
        self.target_model_runner.set_block_size(
            self.target_cache_engine.block_size)
        self.draft_model_runner.set_block_size(
            self.draft_cache_engine.block_size)

    @torch.inference_mode()
    def execute_target_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.target_cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.target_cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.target_cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.target_cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        assert not seq_group_metadata_list[0].sps_stage == SpSStage.DRAFT_DECODE
        output = self.target_model_runner.execute_model(seq_group_metadata_list,
                                                        self.target_gpu_cache, cache_events)
        return output

    @torch.inference_mode()
    def execute_draft_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.draft_cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.draft_cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.draft_cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.draft_cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        assert not seq_group_metadata_list[0].sps_stage == SpSStage.TARGET_DECODE
        output = self.draft_model_runner.execute_model(seq_group_metadata_list,
                                                       self.draft_gpu_cache, cache_events)
        return output

    def _init_distributed_environment(
        self
    ) -> None:
        """Initialize the distributed environment."""
        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != self.parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {self.parallel_config.world_size}).")
        elif not self.distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.parallel_config.world_size,
                rank=self.rank,
                init_method=self.distributed_init_method,
            )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        self.target_parallel_state.initialize_model_parallel(self.parallel_config.tensor_parallel_size,
                                                             self.parallel_config.pipeline_parallel_size)
        self.draft_parallel_state.initialize_model_parallel(self.parallel_config.tensor_parallel_size,
                                                            self.parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
