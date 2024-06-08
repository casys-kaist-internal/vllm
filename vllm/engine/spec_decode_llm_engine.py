import copy
from collections import defaultdict
import os
import time
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Union)
import torch
import torch.multiprocessing as mp

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpecDecodeConfig)
from vllm.core.spec_decode_scheduler import SpecDecodeScheduler, SpecDecodeSchedulerOutputs
from vllm.engine.arg_utils import SpecDecodeEngineArgs
from vllm.engine.metrics import record_metrics
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.engine.mp_utils import init_worker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus, SpecDecodeStage)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter, set_cuda_visible_devices, get_ip, get_open_port, nvtx_range
from vllm.worker.spec_decode_worker import SpecDecodeWorker


if ray:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 1

# Set the start method to 'spawn'
mp.set_start_method('spawn', force=True)


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

        self._init_workers_mp()

        # Draft_probs that should be sent to target worker when target decode
        self.draft_probs_dict = defaultdict(list)

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

    def _init_workers_mp(self):
        self.draft_worker = SpecDecodeWorker(
            copy.deepcopy(self.draft_model_config),
            copy.deepcopy(self.parallel_config),
            copy.deepcopy(self.scheduler_config),
            copy.deepcopy(self.spec_decode_config),
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://{get_ip()}:{get_open_port()}",
        )

        self.draft_worker.init_model()
        self.draft_worker.load_model()

        parent_conn, child_conn = mp.Pipe()

        process = mp.Process(target=init_worker, args=(child_conn, self.target_model_config,
                             self.parallel_config, self.scheduler_config, self.spec_decode_config))
        process.start()

        self.target_worker_process = process
        self.target_worker_pipe = parent_conn

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
        sample = outputs.samples[0]
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

        if spec_decode_stage == SpecDecodeStage.PROMPT:
            seq.append_token_id(sample.output_token,
                                sample.logprobs)
            check_stop_cnt = 1
            num_tokens_to_log_system_stats += (seq.get_len())

        elif spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
            seq.append_draft_token_id(
                sample.output_token, sample.logprobs
            )
            check_stop_cnt = 0
            self.draft_probs_dict[seq.seq_id].append(
                sample.draft_probs)

        elif spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
            free_block_cnt = seq.accept_draft_tokens(
                sample.accept_cnt
            )
            self.scheduler.block_manager.free_blocks(
                seq, free_block_cnt
            )
            # modified_rejection token for not all accept case and bonus token for all accept case
            seq.append_token_id(
                sample.output_token, sample.logprobs
            )
            # If all accept, the bonus token don't have draft kv cache yet.
            if seq.draft_size != sample.accept_cnt:
                seq.data.draft_kv_cache_cnt += 1

            check_stop_cnt = (sample.accept_cnt + 1)
            self.draft_probs_dict[seq.seq_id].clear()
            num_tokens_to_log_system_stats += (sample.accept_cnt + 1)

        else:
            raise ValueError(
                f"Invalid SpecDecodeStage: {spec_decode_stage}")

        if spec_decode_stage != SpecDecodeStage.DRAFT_DECODE:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(
                seq, seq_group.sampling_params, check_stop_cnt)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            if seq.is_finished():
                self.scheduler.free_seq(seq)

        return num_tokens_to_log_system_stats

    @nvtx_range("_process_model_outputs")
    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SpecDecodeSchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        num_prompt_tokens_to_log = 0
        num_generation_tokens_to_log = 0

        for seq_group, outputs in zip(scheduled_seq_groups, output):
            num_tokens_to_log = self._process_sequence_group_outputs(
                seq_group, outputs, seq_group.spec_decode_stage)

            if seq_group.spec_decode_stage == SpecDecodeStage.PROMPT:
                num_prompt_tokens_to_log += num_tokens_to_log
            else:
                num_generation_tokens_to_log += num_tokens_to_log

            if seq_group.spec_decode_stage != SpecDecodeStage.DRAFT_DECODE:
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
                num_prompt_tokens_to_log, num_generation_tokens_to_log)

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

        if len(draft_probs_list) == 0:
            return torch.tensor([], device='cuda').share_memory_()

        return torch.stack(draft_probs_list).share_memory_()

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
            return self._process_model_outputs([], scheduler_outputs)

        # Execute the model.
        if spec_decode_stage == SpecDecodeStage.PROMPT:
            # Execute the target model in the ray process
            output = self._run_target_worker(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
            )

            result = self._process_model_outputs(output, scheduler_outputs)

        elif spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
            if self.spec_decode_config.draft_size == 0:
                return self._process_model_outputs([], scheduler_outputs)

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

                result = self._process_model_outputs(output, scheduler_outputs)

        elif spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
            # Make draft_probs tensor
            draft_probs_tensor = self._make_draft_probs_tensor(
                seq_group_metadata_list)

            # Execute the target model in the different process
            output = self._run_target_worker(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                draft_probs_tensor=draft_probs_tensor
            )

            result = self._process_model_outputs(output, scheduler_outputs)

        else:
            raise ValueError(
                f"Invalid SpecDecodeStage: {spec_decode_stage}")

        return result

    @nvtx_range("collocate_step")
    def collocate_step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        (target_seq_group_metadata_list, draft_seq_group_metadata_list,
         target_scheduler_outputs, draft_scheduler_outputs) = self.scheduler.collocate_schedule()

        if target_scheduler_outputs.is_empty():
            target_result = self._process_model_outputs(
                [], target_scheduler_outputs)
        else:
            # Make shared draft_probs tensor
            draft_probs_tensor = self._make_draft_probs_tensor(
                target_seq_group_metadata_list)

            self._send_target_worker_task(
                "execute_model",
                seq_group_metadata_list=target_seq_group_metadata_list,
                blocks_to_swap_in=target_scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=target_scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=target_scheduler_outputs.blocks_to_copy,
                draft_probs_tensor=draft_probs_tensor
            )

        if draft_scheduler_outputs.is_empty() or self.spec_decode_config.draft_size == 0:
            draft_result = self._process_model_outputs(
                [], draft_scheduler_outputs)
        else:
            for _ in range(self.spec_decode_config.draft_size):
                # Execute the draft model in the same process
                draft_output = self._run_draft_worker(
                    "execute_model",
                    seq_group_metadata_list=draft_seq_group_metadata_list,
                    blocks_to_swap_in=draft_scheduler_outputs.blocks_to_swap_in,
                    blocks_to_swap_out=draft_scheduler_outputs.blocks_to_swap_out,
                    blocks_to_copy=draft_scheduler_outputs.blocks_to_copy,
                )
                draft_scheduler_outputs.blocks_to_swap_in = {}
                draft_scheduler_outputs.blocks_to_swap_out = {}
                draft_scheduler_outputs.blocks_to_copy = {}

                draft_result = self._process_model_outputs(
                    draft_output, draft_scheduler_outputs)

        if not target_scheduler_outputs.is_empty():
            # Wait for the target worker output
            target_output = self._receive_target_worker_output()
            target_result = self._process_model_outputs(
                target_output, target_scheduler_outputs)
            del draft_probs_tensor

        result = target_result + draft_result

        return result

    def _log_system_stats(
        self,
        num_prompt_tokens_to_log: int,
        num_generation_tokens_to_log: int,
    ) -> None:
        # Do not log the stats during the draft stage. Only log final accepted tokens
        if num_prompt_tokens_to_log == 0 and num_generation_tokens_to_log == 0:
            return

        now = time.monotonic()
        # Log the number of batched input tokens.
        self.num_prompt_tokens.append((now, num_prompt_tokens_to_log))
        self.num_generation_tokens.append((now, num_generation_tokens_to_log))

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
            scheduler_running=self.scheduler.get_num_running_seq_groups(),
            scheduler_swapped=len(self.scheduler.swapped),
            scheduler_waiting=len(self.scheduler.waiting),
            gpu_cache_usage=gpu_cache_usage,
            cpu_cache_usage=cpu_cache_usage,
        )

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {self.scheduler.get_num_running_seq_groups()} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    @ nvtx_range("_decode_sequence")
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

    @nvtx_range("_run_target_worker")
    def _run_target_worker(self, method: str, *args, **kwargs) -> Any:
        self.target_worker_pipe.send((method, args, kwargs))
        result = self.target_worker_pipe.recv()
        return result

    def _send_target_worker_task(self, method: str, *args, **kwargs) -> Any:
        self.target_worker_pipe.send((method, args, kwargs))

    def _receive_target_worker_output(self):
        return self.target_worker_pipe.recv()

    @nvtx_range("_run_draft_worker")
    def _run_draft_worker(self, method: str, *args, **kwargs) -> Any:
        worker_instance = self.draft_worker
        return getattr(worker_instance, method)(*args, **kwargs)

    def shutdown(self):
        self.target_worker_pipe.send(("shutdown", [], {}))
        self.target_worker_process.join()
