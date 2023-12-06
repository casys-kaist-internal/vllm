"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpSConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import ParallelState
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
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
        target_parallel_config: ParallelConfig,
        draft_parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        sps_config: SpSConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.target_model_config = target_model_config
        self.draft_model_config = draft_model_config
        self.target_parallel_config = target_parallel_config
        self.draft_parallel_config = draft_parallel_config
        self.scheduler_config = scheduler_config
        self.sps_config = sps_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

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
        self.target_model = get_model(
            self.target_model_config, self.target_parallel_state)
        self.draft_model = get_model(
            self.draft_model_config, self.draft_parallel_state)

        # NOTE(sjchoi): Target and draft should have same vocab size for
        # speculative sampling to work.
        assert self.target_model.config.vocab_size == self.draft_model.config.vocab_size

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

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.target_model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
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
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        # Profiling memory usage should be done with executing
        # both target model and draft model.
        target_num_layers = self.target_model_config.get_num_layers(
            self.target_parallel_config)
        self.target_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * target_num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        draft_num_layers = self.draft_model_config.get_num_layers(
            self.draft_parallel_config)
        self.draft_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * draft_num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        target_cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.target_model_config, self.target_parallel_config)
        draft_cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.draft_model_config, self.draft_parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            (target_cache_block_size + draft_cache_block_size))
        num_cpu_blocks = int(
            cpu_swap_space // (target_cache_block_size + draft_cache_block_size))
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.target_model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.sliding_window = cache_config.sliding_window

        self.target_cache_engine = CacheEngine(self.cache_config, self.target_model_config,
                                               self.target_parallel_config)
        self.draft_cache_engine = CacheEngine(self.cache_config, self.draft_model_config,
                                              self.draft_parallel_config)

        self.target_cache_events = self.target_cache_engine.events
        self.draft_cache_events = self.draft_cache_engine.events

        self.target_gpu_cache = self.target_cache_engine.gpu_cache
        self.draft_gpu_cache = self.draft_cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            if sampling_params.prompt_logprobs is not None:
                # NOTE: prompt token positions do not need sample, skip
                categorized_sample_indices_start_idx += prompt_len - 1

            categorized_sample_indices[sampling_params.sampling_type].append(
                categorized_sample_indices_start_idx)
            categorized_sample_indices_start_idx += 1

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([0] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        max_seq_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_metadata.is_prompt:
                # We need to do this in this loop as we need to know max_seq_len
                assert len(
                    seq_ids) == 1, "Prompt input should have only one seq."
                sampling_params = seq_group_metadata.sampling_params
                assert len(prompt_lens) == len(seq_group_metadata_list)
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_seq_len
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            num_seqs = len(seq_ids)
            selected_token_indices.extend(
                range(selected_token_start_idx,
                      selected_token_start_idx + num_seqs))
            selected_token_start_idx += num_seqs

            categorized_sample_indices[sampling_params.sampling_type].extend(
                range(categorized_sample_indices_start_idx,
                      categorized_sample_indices_start_idx + num_seqs))
            categorized_sample_indices_start_idx += num_seqs

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                context_len = seq_data.get_len()
                position = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append([position])

                block_table = seq_group_metadata.block_tables[seq_id]

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)

        padded_input_tokens = [
            _pad_to_max(tokens, max_seq_len, pad=0) for tokens in input_tokens
        ]
        padded_input_positions = [
            _pad_to_max(positions, max_seq_len, pad=0)
            for positions in input_positions
        ]
        padded_slot_mapping = [
            _pad_to_max(mapping, max_seq_len, pad=-1)
            for mapping in slot_mapping
        ]
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq, pad=0)
            for block_table in generation_block_tables
        ]

        # Convert to tensors.
        tokens_tensor = torch.tensor(padded_input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(padded_input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensor = torch.tensor(padded_slot_mapping,
                                           dtype=torch.long,
                                           device="cuda")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cuda")
        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="cuda")
        categorized_sample_indices = {
            t: torch.tensor(seq_ids, dtype=torch.int, device="cuda")
            for t, seq_ids in categorized_sample_indices.items()
        }
        block_tables_tensor = torch.tensor(padded_block_tables,
                                           dtype=torch.int,
                                           device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            sliding_window=self.sliding_window,
        )
        return tokens_tensor, positions_tensor, input_metadata

    def _prepare_draft_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            if sampling_params.prompt_logprobs is not None:
                # NOTE: prompt token positions do not need sample, skip
                categorized_sample_indices_start_idx += prompt_len - 1

            categorized_sample_indices[sampling_params.sampling_type].append(
                categorized_sample_indices_start_idx)
            categorized_sample_indices_start_idx += 1

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([0] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        max_seq_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_metadata.is_prompt:
                # We need to do this in this loop as we need to know max_seq_len
                assert len(
                    seq_ids) == 1, "Prompt input should have only one seq."
                sampling_params = seq_group_metadata.sampling_params
                assert len(prompt_lens) == len(seq_group_metadata_list)
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_seq_len
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            num_seqs = len(seq_ids)
            selected_token_indices.extend(
                range(selected_token_start_idx,
                      selected_token_start_idx + num_seqs))
            selected_token_start_idx += num_seqs

            categorized_sample_indices[sampling_params.sampling_type].extend(
                range(categorized_sample_indices_start_idx,
                      categorized_sample_indices_start_idx + num_seqs))
            categorized_sample_indices_start_idx += num_seqs

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                draft_token = seq_data.get_last_draft_token_id()
                input_tokens.append([draft_token])

                context_len = seq_data.get_len() + seq_data.get_draft_len()
                position = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append([position])

                block_table = seq_group_metadata.block_tables[seq_id]

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)

        padded_input_tokens = [
            _pad_to_max(tokens, max_seq_len, pad=0) for tokens in input_tokens
        ]
        padded_input_positions = [
            _pad_to_max(positions, max_seq_len, pad=0)
            for positions in input_positions
        ]
        padded_slot_mapping = [
            _pad_to_max(mapping, max_seq_len, pad=-1)
            for mapping in slot_mapping
        ]
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq, pad=0)
            for block_table in generation_block_tables
        ]

        # Convert to tensors.
        tokens_tensor = torch.tensor(padded_input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(padded_input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensor = torch.tensor(padded_slot_mapping,
                                           dtype=torch.long,
                                           device="cuda")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cuda")
        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="cuda")
        categorized_sample_indices = {
            t: torch.tensor(seq_ids, dtype=torch.int, device="cuda")
            for t, seq_ids in categorized_sample_indices.items()
        }
        block_tables_tensor = torch.tensor(padded_block_tables,
                                           dtype=torch.int,
                                           device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            sliding_window=self.sliding_window,
        )
        return tokens_tensor, positions_tensor, input_metadata

    def _prepare_target_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0
        draft_lens: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            if sampling_params.prompt_logprobs is not None:
                # NOTE: prompt token positions do not need sample, skip
                categorized_sample_indices_start_idx += prompt_len - 1

            categorized_sample_indices[sampling_params.sampling_type].append(
                categorized_sample_indices_start_idx)
            categorized_sample_indices_start_idx += 1

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([0] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        max_seq_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            if seq_group_metadata.is_prompt:
                # We need to do this in this loop as we need to know max_seq_len
                assert len(
                    seq_ids) == 1, "Prompt input should have only one seq."
                sampling_params = seq_group_metadata.sampling_params
                assert len(prompt_lens) == len(seq_group_metadata_list)
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_seq_len
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            num_seqs = len(seq_ids)
            selected_token_indices.extend(
                range(selected_token_start_idx,
                      selected_token_start_idx + num_seqs))
            selected_token_start_idx += num_seqs

            categorized_sample_indices[sampling_params.sampling_type].extend(
                range(categorized_sample_indices_start_idx,
                      categorized_sample_indices_start_idx + num_seqs))
            categorized_sample_indices_start_idx += num_seqs

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                draft_tokens = [seq_data.get_last_token_id()] + \
                    seq_data.get_draft_token_ids()
                draft_lens.append(len(draft_tokens))
                input_tokens.append([draft_tokens])

                context_len = seq_data.get_len()
                position_start = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append(
                    [range(position_start, position_start + len(draft_tokens))])

                block_table = seq_group_metadata.block_tables[seq_id]

                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))

                # Compoute context length
                for i in range(len(draft_tokens)):
                    context_lens.append(context_len + i)
                max_context_len = max(max_context_len, max(context_lens))

                # Compute the slot mapping
                for i in range(position_start, position_start + len(draft_tokens)):
                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)

        padded_input_tokens = [
            _pad_to_max(tokens, max_seq_len, pad=0) for tokens in input_tokens
        ]
        padded_input_positions = [
            _pad_to_max(positions, max_seq_len, pad=0)
            for positions in input_positions
        ]
        padded_slot_mapping = [
            _pad_to_max(mapping, max_seq_len, pad=-1)
            for mapping in slot_mapping
        ]
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq, pad=0)
            for block_table in generation_block_tables
        ]

        # Convert to tensors.
        tokens_tensor = torch.tensor(padded_input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(padded_input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensor = torch.tensor(padded_slot_mapping,
                                           dtype=torch.long,
                                           device="cuda")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cuda")
        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="cuda")
        categorized_sample_indices = {
            t: torch.tensor(seq_ids, dtype=torch.int, device="cuda")
            for t, seq_ids in categorized_sample_indices.items()
        }
        block_tables_tensor = torch.tensor(padded_block_tables,
                                           dtype=torch.int,
                                           device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            sliding_window=self.sliding_window,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output

    @torch.inference_mode()
    def execute_target_model_for_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.target_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.target_gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output

    @torch.inference_mode()
    def execute_draft_model_for_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.draft_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.target_gpu_cache,  # is this right??
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
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
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_draft_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.draft_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.draft_gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output

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
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # Prepare input tensors.
        input_tokens, input_positions, input_metadata = self._prepare_target_inputs(
            seq_group_metadata_list)

        # Execute the model.
        output = self.target_model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.target_gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output

    def _init_distributed_environment(
        self
    ) -> None:
        """Initialize the distributed environment."""
        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != self.target_parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch world "
                    "size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {self.target_parallel_config.world_size}).")
        elif not self.distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=self.target_parallel_config.world_size,
                rank=self.rank,
                init_method=self.distributed_init_method,
            )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        self.target_parallel_state.initialize_model_parallel(self.target_parallel_config.tensor_parallel_size,
                                                             self.target_parallel_config.pipeline_parallel_size)
        self.draft_parallel_state.initialize_model_parallel(self.draft_parallel_config.tensor_parallel_size,
                                                            self.draft_parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int, pad: int) -> List[int]:
    return x + [pad] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return x + [pad] * (max_len - len(x))


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
