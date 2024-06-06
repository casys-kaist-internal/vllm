import copy
from collections import defaultdict
import os
import time
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Union)
import torch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpecDecodeConfig)
from vllm.core.spec_decode_scheduler import SpecDecodeScheduler, SpecDecodeSchedulerOutputs
from vllm.engine.arg_utils import SpecDecodeEngineArgs
from vllm.engine.metrics import record_metrics
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus, SpecDecodeStage)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter, set_cuda_visible_devices, get_ip, get_open_port, nvtx_range

if ray:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class SpecDecodeLLMEngine:
    """An LLM engine that receives requests and generates texts.

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
        spec_decode_config: SpecDecodeConfig,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={target_model_config.model!r}, "
            f"tokenizer={draft_model_config.tokenizer!r}, "
            f"tokenizer_mode={target_model_config.tokenizer_mode}, "
            f"revision={target_model_config.revision}, "
            f"tokenizer_revision={target_model_config.tokenizer_revision}, "
            f"trust_remote_code={target_model_config.trust_remote_code}, "
            f"dtype={target_model_config.dtype}, "
            f"max_seq_len={target_model_config.max_model_len}, "
            f"download_dir={target_model_config.download_dir!r}, "
            f"load_format={target_model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"quantization={target_model_config.quantization}, "
            f"enforce_eager={target_model_config.enforce_eager}, "
            f"seed={target_model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.target_model_config = target_model_config
        self.draft_model_config = draft_model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.spec_decode_config = spec_decode_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            target_model_config.tokenizer,
            tokenizer_mode=target_model_config.tokenizer_mode,
            trust_remote_code=target_model_config.trust_remote_code,
            tokenizer_revision=target_model_config.tokenizer_revision,
            revision=target_model_config.revision)
        self.seq_counter = Counter()

        assert self.parallel_config.worker_use_ray

        # Create the parallel GPU workers.
        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        self._init_workers_ray(placement_group)
        ray.put("dummy")

        # Draft_probs that should be sent to target worker when target decode
        self.draft_probs_dict = defaultdict(list)
        self.draft_probs_tensor = torch.zeros(64, 7, 50272)
        self.draft_probs_tensor_ref = ray.put(self.draft_probs_tensor)

        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = SpecDecodeScheduler(
            scheduler_config, cache_config, spec_decode_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def _configure_ray_workers_use_nsight(self,
                                          ray_remote_kwargs) -> Dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        runtime_env.update({
            "nsight": {
                "t": "cuda,cudnn,cublas,nvtx",
                "gpu-metrics-device": "0",
                "cuda-graph-trace": "node",
            }
        })

        return ray_remote_kwargs

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        assert self.parallel_config.tensor_parallel_size == 1
        num_gpus = 0.5  # Since there will be two workers, we use 0.5 GPU per worker

        assert len(
            placement_group.bundle_specs) == 1, ("We only consider single GPU for now.")

        if self.parallel_config.ray_workers_use_nsight:
            print("Using nsight profiling for Ray workers.")
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        driver_ip = get_ip()
        bundle_id = 0

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_id,
        )

        worker = ray.remote(
            num_cpus=0,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            **ray_remote_kwargs,
        )(RayWorkerVllm).remote(self.target_model_config.trust_remote_code)

        self.target_worker: RayWorkerVllm = worker

        if self.target_worker is None:
            raise ValueError(
                "Ray does not allocate any GPUs on the driver node. Consider "
                "adjusting the Ray placement group or running the driver on a "
                "GPU node.")

        _, driver_gpu_ids = ray.get(
            self.target_worker.get_node_and_gpu_ids.remote())

        # Set CUDA_VISIBLE_DEVICES for the driver
        set_cuda_visible_devices(driver_gpu_ids)

        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.spec_decode_worker import SpecDecodeWorker

        # Initialize torch distributed process group for the workers.
        target_model_config = copy.deepcopy(self.target_model_config)
        draft_model_config = copy.deepcopy(self.draft_model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        spec_decode_config = copy.deepcopy(self.spec_decode_config)

        driver_rank = 0
        driver_local_rank = 0

        # Driver worker
        distributed_init_method = f"tcp://{driver_ip}:{get_open_port()}"
        self.draft_worker = SpecDecodeWorker(
            draft_model_config,
            parallel_config,
            scheduler_config,
            spec_decode_config,
            driver_local_rank,
            driver_rank,
            distributed_init_method,
            is_driver_worker=True,
        )

        # Ray worker
        distributed_init_method = f"tcp://{driver_ip}:{get_open_port()}"
        self.target_worker.init_worker.remote(
            lambda rank=driver_rank, local_rank=driver_local_rank: SpecDecodeWorker(
                target_model_config,
                parallel_config,
                scheduler_config,
                spec_decode_config,
                local_rank,
                rank,
                distributed_init_method,
            )
        )

        self._run_draft_worker("init_model")
        self._run_draft_worker("load_model")

        self._run_target_worker("init_model")
        self._run_target_worker("load_model")

    def _init_draft_probs(self, draft_size: int) -> None:
        # draft_probs: [max_seqs, draft_size, vocab_size]
        self.draft_probs = torch.zeros(
            64,
            draft_size,
            self.tokenizer.vocab_size,
            device="cuda",
        )

    def _verify_args(self) -> None:
        self.target_model_config.verify_with_parallel_config(
            self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        initial_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()

        draft_consumed_memory, draft_cache_block_size = self._run_draft_worker(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        target_consumed_memory, target_cache_block_size = self._run_target_worker(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        cache_block_size = draft_cache_block_size + target_cache_block_size
        peak_gpu_memory = total_gpu_memory - (initial_free_gpu_memory -
                                              (draft_consumed_memory + target_consumed_memory))
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_gpu_memory) // cache_block_size)
        num_cpu_blocks = int(
            self.cache_config.swap_space_bytes // cache_block_size)

        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        if self.target_model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.target_model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_draft_worker("init_cache_engine",
                               cache_config=self.cache_config)
        self._run_target_worker("init_cache_engine",
                                cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self._run_draft_worker("warm_up_model")
        self._run_target_worker("warm_up_model")

    @ classmethod
    def from_engine_args(cls, engine_args: SpecDecodeEngineArgs) -> "SpecDecodeLLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[3]
        # Initialize the cluster.
        placement_group = initialize_cluster(parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs,
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
        seq = Sequence(seq_id, prompt, prompt_token_ids,
                       block_size, self.spec_decode_config.draft_size)

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
        """Gets the model configuration."""
        return self.target_model_config

    def get_draft_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.draft_model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id))
        return current_worst_score >= highest_attainable_score

    @nvtx_range("_process_sequence_group_outputs")
    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput,
                                        spec_decode_stage: SpecDecodeStage) -> int:
        # We assume that SpecDecodeEngine does not use beam search
        assert not seq_group.sampling_params.use_beam_search

        num_tokens_to_log_system_stats = 0

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
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue

            assert len(child_samples) == 1

            child_sample = child_samples[0]

            if spec_decode_stage == SpecDecodeStage.PROMPT:
                parent.append_token_id(child_sample.output_token,
                                       child_sample.logprobs)
                check_stop_cnt = 1
                num_tokens_to_log_system_stats += (parent.get_len())

            elif spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
                parent.append_draft_token_id(
                    child_sample.output_token, child_sample.logprobs
                )
                check_stop_cnt = 0
                self.draft_probs_dict[parent.seq_id].append(
                    child_sample.draft_probs)

            elif spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
                free_block_cnt = parent.accept_draft_tokens(
                    child_sample.accept_cnt
                )
                self.scheduler.block_manager.free_blocks(
                    parent, free_block_cnt
                )
                # modified_rejection token for not all accept case and bonus token for all accept case
                parent.append_token_id(
                    child_sample.output_token, child_sample.logprobs
                )
                # If all accept, the bonus token don't have draft kv cache yet.
                if parent.draft_size != child_sample.accept_cnt:
                    parent.data.draft_kv_cache_cnt += 1

                check_stop_cnt = (child_sample.accept_cnt + 1)
                self.draft_probs_dict[parent.seq_id].clear()
                num_tokens_to_log_system_stats += (child_sample.accept_cnt + 1)

            else:
                raise ValueError(
                    f"Invalid SpecDecodeStage: {spec_decode_stage}")

            child_seqs.append((parent, parent, check_stop_cnt))

        if spec_decode_stage != SpecDecodeStage.DRAFT_DECODE:
            for seq, _, check_stop_cnt in child_seqs:
                self._decode_sequence(seq, seq_group.sampling_params)
                self._check_stop(
                    seq, seq_group.sampling_params, check_stop_cnt)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            for seq, parent, _ in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)

        return num_tokens_to_log_system_stats

    @nvtx_range("_process_model_outputs")
    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SpecDecodeSchedulerOutputs,
            spec_decode_stage: SpecDecodeStage) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        num_tokens_to_log_system_stats = 0

        for seq_group, outputs in zip(scheduled_seq_groups, output):
            num_tokens_to_log_system_stats += self._process_sequence_group_outputs(
                seq_group, outputs, spec_decode_stage)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(
                spec_decode_stage, num_tokens_to_log_system_stats)

        return request_outputs

    @nvtx_range("_make_draft_probs_tensor")
    def _make_draft_probs_tensor(self,
                                 seq_group_metadata_list: List[SequenceGroupMetadata]) -> torch.Tensor:
        draft_probs_list = []
        for seq_group_metadata in seq_group_metadata_list:
            keys = list(seq_group_metadata.seq_data.keys())
            assert len(keys) == 1
            seq_id = keys[0]
            draft_probs = self.draft_probs_dict[seq_id]
            draft_probs_list.extend(draft_probs)
        draft_probs_tensor = torch.stack(draft_probs_list)
        return draft_probs_tensor

    @nvtx_range("step")
    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        spec_decode_stage = seq_group_metadata_list[0].spec_decode_stage

        if scheduler_outputs.is_empty():
            return self._process_model_outputs([], scheduler_outputs, spec_decode_stage)

        # Execute the model.
        if spec_decode_stage == SpecDecodeStage.PROMPT:
            # Execute the target model in the ray process
            seq_group_metadata_list_ref = ray.put(
                seq_group_metadata_list)

            output = self._run_target_worker(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list_ref,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )

            result = self._process_model_outputs(
                output, scheduler_outputs, spec_decode_stage)

        elif spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
            for _ in range(self.spec_decode_config.draft_size):
                # Execute the draft model in the same process
                output = self._run_draft_worker(
                    "execute_model",
                    seq_group_metadata_list=seq_group_metadata_list,
                    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                    blocks_to_copy=scheduler_outputs.blocks_to_copy,
                )
                scheduler_outputs.blocks_to_swap_in = {}
                scheduler_outputs.blocks_to_swap_out = {}
                scheduler_outputs.blocks_to_copy = {}

                result = self._process_model_outputs(
                    output, scheduler_outputs, spec_decode_stage)

        elif spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
            # Make draft_probs tensor
            draft_probs_tensor = self._make_draft_probs_tensor(
                seq_group_metadata_list)

            torch.cuda.nvtx.range_push("metadata put")
            seq_group_metadata_list_ref = ray.put(
                seq_group_metadata_list)
            torch.cuda.nvtx.range_pop()

            # Execute the target model in the ray process
            output = self._run_target_worker(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list_ref,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                draft_probs_tensor=draft_probs_tensor
            )

            result = self._process_model_outputs(
                output, scheduler_outputs, spec_decode_stage)

        else:
            raise ValueError(
                f"Invalid SpecDecodeStage: {spec_decode_stage}")

        return result

    def _log_system_stats(
        self,
        spec_decode_stage: SpecDecodeStage,
        num_batched_tokens: int,
    ) -> None:
        # Do not log the stats during the draft stage. Only log final accepted tokens
        if spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
            return

        now = time.monotonic()
        # Log the number of batched input tokens.
        if spec_decode_stage == SpecDecodeStage.PROMPT:
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
                    check_stop_cnt: int) -> None:
        """Stop the finished sequences."""
        # We dont support sampling_params stop string in Spec Decode
        last_tokens = seq.get_last_token_ids(check_stop_cnt)

        if any(token in sampling_params.stop_token_ids for token in last_tokens):
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
        if ((not sampling_params.ignore_eos)
                and any(token == self.tokenizer.eos_token_id for token in last_tokens)):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    @ nvtx_range("_run_draft_worker")
    def _run_draft_worker(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on draft worker"""
        draft_worker_output = getattr(self.draft_worker,
                                      method)(*args, **kwargs)

        return draft_worker_output

    @ nvtx_range("_run_target_worker")
    def _run_target_worker(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on target worker"""
        # Start the ray worker.
        target_worker_output = self.target_worker.execute_method.remote(
            method, *args, **kwargs)

        # Get the results of the ray worker
        target_worker_output = ray.get(target_worker_output)

        return target_worker_output
