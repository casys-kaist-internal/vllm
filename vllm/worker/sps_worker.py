"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed
from tabulate import tabulate

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
                                                  scheduler_config, sps_config)
        self.draft_model_runner = SpSModelRunner(draft_model_config, parallel_config,
                                                 scheduler_config, sps_config)

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

        # Create a list of lists to store the data for the table
        table_data = [
            ["gpu_mem_util (%)", gpu_memory_utilization * 100],
            ["total_gpu_memory (GiB)", format(
                total_gpu_memory / 1024 ** 3, ".1f")],
            ["peak_memory (GiB)", format(peak_memory / 1024 ** 3, ".1f")],
            ["target_cache_block_size (KiB)", target_cache_block_size / 1024],
            ["draft_cache_block_size (KiB)", draft_cache_block_size / 1024],
            ["block_size", block_size],
            ["num_gpu_blocks", num_gpu_blocks]
        ]

        # Print the table using the tabulate function
        print(tabulate(table_data, headers=[
              "Parameter", "Value"], tablefmt="grid"))

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
    
    
    def _process_draft_sequence_group_outputs(self, seq_group: SequenceGroup,
                                                outputs: SequenceGroupOutput,
                                                sps_stage: SpSStage) -> None:

        # We assume that SpS engine does not use beam search.
        assert not seq_group.sampling_params.use_beam_search

        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutput] = parent_child_dict[
                parent.seq_id]
            assert len(child_samples) == 1, (
                "SpS engine does not use beam search, so there should be "
                "exactly one child sample for each parent sequence.")

            child_sample = child_samples[0]
            check_cnt = 1


            if sps_stage == SpSStage.DRAFT_DECODE:
                # Append the draft token to the parent sequence.
                parent.append_draft_token_id(child_sample.output_token,
                                             child_sample.logprobs,
                                             child_sample.probs)

            else:
                raise ValueError(f"Invalid SpS stage: {sps_stage}")

            child_seqs.append((parent, parent, check_cnt))

        # Decode and check stop only on PROMPT stage and TARGET_DECODE stage.
        if sps_stage == SpSStage.PROMPT or sps_stage == SpSStage.TARGET_DECODE:
            for seq, _, check_cnt in child_seqs:
                self._decode_sequence(seq, seq_group.sampling_params)
                self._check_stop(seq, seq_group.sampling_params, check_cnt)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        # NOTE: we need to fork the new sequences before freeing the
        # old sequences.
        for seq, parent, _ in child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)
        return
    
    def _process_draft_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SpSSchedulerOutputs,
            sps_stage: SpSStage) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            self._process_draft_sequence_group_outputs(seq_group, outputs, sps_stage)


        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(True if sps_stage == SpSStage.PROMPT else False,
                                   scheduler_outputs.num_batched_tokens)
        return request_outputs
    
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
    
    
    @torch.inference_mode()
    def execute_multi_step_draft_model(
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

        # run this several times 
        for _  in range(3):
            output = self.draft_model_runner.execute_model(seq_group_metadata_list,
                                                        self.draft_gpu_cache, cache_events)
            self._process_draft_model_outputs(output, scheduler_outputs, SpSStage.DRAFT_DECODE)

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
