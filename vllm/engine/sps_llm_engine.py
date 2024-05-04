import os
import copy
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, Dict
import itertools

import torch
from torch.cuda import nvtx

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpSConfig)
from vllm.core.sps_scheduler import SpSScheduler, SpSSchedulerOutputs
from vllm.engine.arg_utils import SpSEngineArgs
from vllm.engine.metrics import record_metrics
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import SpSRequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus, SpSStage)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter

if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class SpSLLMEngine:
    """A SpS LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        target_model_config: ModelConfig,
        draft_model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        sps_config: SpSConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an SpS LLM engine with config: "
            f"target_model={target_model_config.model!r}, "
            f"draft_model={draft_model_config.model!r}, "
            f"tokenizer={target_model_config.tokenizer!r}, "
            f"tokenizer_mode={target_model_config.tokenizer_mode}, "
            f"revision={target_model_config.revision}, "
            f"tokenizer_revision={target_model_config.tokenizer_revision}, "
            f"trust_remote_code={target_model_config.trust_remote_code}, "
            f"dtype={target_model_config.dtype}, "
            f"max_seq_len={target_model_config.max_model_len}, "
            f"target_download_dir={target_model_config.download_dir!r}, "
            f"draft_download_dir={draft_model_config.download_dir!r}, "
            f"load_format={target_model_config.load_format}, "
            f"target_tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"target_model_quantization={target_model_config.quantization}, "
            f"draft_model_quantization={target_model_config.quantization}, "
            f"seed={target_model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.target_model_config = target_model_config
        self.draft_model_config = draft_model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.sps_config = sps_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            target_model_config.tokenizer,
            tokenizer_mode=target_model_config.tokenizer_mode,
            trust_remote_code=target_model_config.trust_remote_code,
            tokenizer_revision=target_model_config.tokenizer_revision,
            revision=target_model_config.revision)
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = SpSScheduler(
            scheduler_config, cache_config, sps_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _init_workers(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.sps_worker import SpSWorker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if target_parallel_config.world_size > 1.")

        self.workers: List[SpSWorker] = []
        worker = SpSWorker(
            self.target_model_config,
            self.draft_model_config,
            self.parallel_config,
            self.scheduler_config,
            self.sps_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self._run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.sps_worker import SpSWorker

        self.workers: List[SpSWorker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            if self.parallel_config.tensor_parallel_size == 1:
                num_gpus = self.cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorkerVllm).remote(self.target_model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        target_model_config = copy.deepcopy(self.target_model_config)
        draft_model_config = copy.deepcopy(self.draft_model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        sps_config = copy.deepcopy(self.sps_config)
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: SpSWorker(
                              target_model_config,
                              draft_model_config,
                              parallel_config,
                              scheduler_config,
                              sps_config,
                              None,
                              None,
                          ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self._run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

    def _verify_args(self) -> None:
        # NOTE(sjchoi): No need to verify draft model since draft model is
        # copied to every GPU.
        self.target_model_config.verify_with_parallel_config(
            self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: SpSEngineArgs) -> "SpSLLMEngine":
        """Creates an SpS LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[3]  # parallel_config for target model
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, self.sps_config.draft_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def abort_all_requests(self) -> None:
        """Aborts all requests."""
        self.scheduler.abort_all_seq_groups()

    def get_target_model_config(self) -> ModelConfig:
        """Gets the target model configuration."""
        return self.target_model_config

    def get_draft_model_config(self) -> ModelConfig:
        """Gets the draft model configuration."""
        return self.draft_model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(
        self
    ) -> Tuple[List[SequenceGroupMetadata], SpSSchedulerOutputs,
               List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
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

            if sps_stage == SpSStage.PROMPT:
                # Append the token to the parent sequence.
                parent.append_token_id(child_sample.output_token,
                                       child_sample.logprobs)

            elif sps_stage == SpSStage.DRAFT_DECODE:
                # Append the draft token to the parent sequence.
                # parent.append_draft_token_id(child_sample.output_token,
                #                              child_sample.logprobs,
                #                              child_sample.probs)
                pass

            elif sps_stage == SpSStage.TARGET_DECODE:
                free_block_cnt = parent.accept_draft_tokens(
                    child_sample.accept_cnt, child_sample.accept_probs, child_sample.beta_list)
                self.scheduler.block_manager.free_blocks(
                    parent, free_block_cnt)
                check_cnt += child_sample.accept_cnt

                # print("accept_cnt / total_cnt ",
                #       child_sample.accept_cnt, child_sample.total_cnt)


                if self.sps_config.use_lazy_draft_kv_cache:
                    if child_sample.accept_cnt != child_sample.total_cnt:
                        parent.data.draft_cache_cnt += 1
                    parent.append_token_id(
                        child_sample.output_token, child_sample.logprobs)
                    check_cnt += 1  # Need to fix this 
                    
                else: 
                    # All accepted, sample additional token and save the bonus token to be appended 
                    # after the draft kv cache is updated
                    parent.save_bonus_token_id(child_sample.output_token, child_sample.logprobs)
                    parent.status = SequenceStatus.SPS_ALL_ACCEPT

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

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SpSSchedulerOutputs,
            sps_stage: SpSStage) -> List[RequestOutput]:
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups

        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        if sps_stage != SpSStage.DRAFT_DECODE:
            for seq_group, outputs in zip(scheduled_seq_groups, output):
                self._process_sequence_group_outputs(seq_group, outputs, sps_stage)

            # Free the finished sequence groups.
            self.scheduler.free_finished_seq_groups()

        # Update the draft target queues.
        self.scheduler.update_draft_target_queues(sps_stage)

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
    
    def step(self) -> List[RequestOutput]:
        """ Performs one decoding iteration: 
            - Target prompt + Draft prompt 
            - Draft decode * multi-step
            - Target decode 
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        sps_stage = seq_group_metadata_list[0].sps_stage
        if sps_stage == SpSStage.PROMPT:
            output = self._run_workers(
                "execute_target_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )

            if not self.sps_config.use_lazy_draft_kv_cache:
                # Don't need output for draft model. Execution required for draft KV cache.
                self._run_workers(
                    "execute_draft_model",
                    seq_group_metadata_list=seq_group_metadata_list,
                    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                    blocks_to_copy=scheduler_outputs.blocks_to_copy,
                )

            return self._process_model_outputs(output, scheduler_outputs, sps_stage), sps_stage
            
        elif sps_stage == SpSStage.DRAFT_DECODE:
            output = self._run_workers(
                "execute_multi_step_draft_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )

            return self._process_model_outputs(output, scheduler_outputs, sps_stage), sps_stage

        elif sps_stage == SpSStage.TARGET_DECODE:
            output = self._run_workers(
                "execute_target_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )

            target_output = self._process_model_outputs(output, scheduler_outputs, sps_stage)

            if not self.sps_config.use_lazy_draft_kv_cache:
                # For seqs that are all accepted, need to run draft model to cache kv for the 
                # additional bonus token sampled by target model
                seq_group_metadata_list: List[SequenceGroupMetadata] = []
                for seq_group in scheduler_outputs.scheduled_seq_groups:
                    seq_data: Dict[int, SequenceData] = {}
                    block_tables: Dict[int, List[int]] = {}
                    for seq in seq_group.get_seqs(status=SequenceStatus.SPS_ALL_ACCEPT):
                        seq_id = seq.seq_id
                        seq_data[seq_id] = seq.data
                        block_tables[seq_id] = self.scheduler.block_manager.get_block_table(
                            seq)
                        
                    if seq_data:
                        seq_group_metadata = SequenceGroupMetadata(
                            request_id=seq_group.request_id,
                            is_prompt=False,
                            seq_data=seq_data,
                            sampling_params=seq_group.sampling_params,
                            block_tables=block_tables,
                            sps_stage=SpSStage.DRAFT_DECODE,
                        )
                        seq_group_metadata_list.append(seq_group_metadata)
                    
                if seq_group_metadata_list:
                    self._run_workers(
                        "execute_draft_model",
                        seq_group_metadata_list=seq_group_metadata_list,
                        blocks_to_swap_in=None,
                        blocks_to_swap_out=None,
                        blocks_to_copy=None,
                    )

                    for seq_group in scheduler_outputs.scheduled_seq_groups:
                        for seq in seq_group.get_seqs(status=SequenceStatus.SPS_ALL_ACCEPT):
                            # change back to original RUNNING status
                            seq.status = SequenceStatus.RUNNING
                            # Append the bonus token saved in all accept case for target decode 
                            seq.append_bonus_token_id()

            return target_output, sps_stage

        else:
            raise ValueError(f"Invalid SpS stage: {sps_stage}")

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        should_log = now - self.last_logging_time >= _LOGGING_INTERVAL_SEC
        if not should_log:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        record_metrics(
            avg_prompt_throughput=avg_prompt_throughput,
            avg_generation_throughput=avg_generation_throughput,
            scheduler_running=len(self.scheduler.running),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset, decode_offset) = detokenize_incrementally(
             self.tokenizer,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             decode_offset=seq.decode_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.decode_offset = decode_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams,
                    check_cnt: int) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            try:
                # Initialize the index to start checking from
                start_idx = 1 - len(stop_str) - check_cnt
                stop_str_idx = seq.output_text.index(stop_str, start_idx)
                seq.output_text = seq.output_text[:stop_str_idx]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
            except ValueError:
                pass

        for idx in range(-check_cnt, 0):
            if seq.get_last_nth_token_id(idx) in sampling_params.stop_token_ids:
                seq.status = SequenceStatus.FINISHED_STOPPED
                return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() >= sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        for idx in range(-check_cnt, 0):
            if ((not sampling_params.ignore_eos)
                    and seq.get_last_nth_token_id(idx) == self.tokenizer.eos_token_id):
                seq.status = SequenceStatus.FINISHED_STOPPED
                return

    def _run_workers_in_batch(
        self,
        workers,
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        return all_outputs

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                self.workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [self.workers]

        for workers in work_groups:
            all_outputs.extend(
                self._run_workers_in_batch(workers, method, *args, **kwargs))

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
