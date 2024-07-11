import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, SpecDecodeConfig
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.sampling_metadata import SamplingTensors

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata, SpecDecodeStage
from vllm.utils import in_wsl, nvtx_range

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class SpecDecodeModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        spec_decode_config: SpecDecodeConfig,
        is_target: bool = True,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.spec_decode_config = spec_decode_config
        self.is_target = is_target

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.model = None
        self.block_size = None  # Set after initial profiling.

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()

    def load_model(self) -> None:
        self.model = get_model(self.model_config)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        # Add draft size to max_context_len_to_capture because the end condition is checked
        # on the target decode stage so draft decode stage can over run the max_context_len_to_capture
        max_num_blocks = (self.max_context_len_to_capture + self.spec_decode_config.draft_size + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_profile_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.spec_decode_stage == SpecDecodeStage.PREFILL
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            single_slot_mapping = []
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prompt_len):
                if i < start_idx:
                    single_slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                single_slot_mapping.append(slot)

            slot_mapping.extend(single_slot_mapping)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device="cuda")
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device="cuda")
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device="cuda")

        input_metadata = InputMetadata(
            num_prefill_tokens=sum(prompt_lens),
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            prefill_lens=prompt_lens,
            target_lens=None,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            use_cuda_graph=False,
            use_target_attention=False,
        )
        return input_tokens, input_positions, input_metadata, prompt_lens

    @nvtx_range("_prepare_prefill")
    def _prepare_prefill(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        prompt_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.spec_decode_stage == SpecDecodeStage.PREFILL
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()

            if self.scheduler_config.chunk_prefill_enabled:
                prompt_tokens[:seq_group_metadata.token_chunk_size]

            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            single_slot_mapping = []
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)

            for i in range(prompt_len):
                if i < start_idx:
                    single_slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                single_slot_mapping.append(slot)

            slot_mapping.extend(single_slot_mapping)

        return input_tokens, input_positions, slot_mapping, prompt_lens

    @nvtx_range("_prepare_target_decode")
    def _prepare_target_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        target_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.spec_decode_stage == SpecDecodeStage.TARGET_DECODE

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_tokens = [
                    seq_data.get_last_token_id()] + seq_data.get_draft_token_ids()
                target_lens.append(len(generation_tokens))
                input_tokens.extend(generation_tokens)

                position_start = seq_data.get_len() - 1
                for i in range(len(generation_tokens)):
                    position = position_start + i
                    input_positions.append(position)

                for i in range(len(generation_tokens)):
                    context_len = seq_data.get_len() + i
                    if self.sliding_window is not None:
                        context_len = min(context_len, self.sliding_window)
                    context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]

                for position in range(position_start, position_start + len(generation_tokens)):
                    block_number = block_table[position // self.block_size]
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping.append(slot)

                    if self.sliding_window is not None:
                        sliding_window_blocks = (self.sliding_window //
                                                 self.block_size)
                        block_table = block_table[-sliding_window_blocks:]
                    block_tables.append(block_table)

        return input_tokens, input_positions, slot_mapping, context_lens, block_tables, target_lens

    @nvtx_range("_prepare_draft_decode")
    def _prepare_draft_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        draft_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.spec_decode_stage == SpecDecodeStage.DRAFT_DECODE

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                num_computed_draft_tokens = seq_data.get_num_computed_draft_tokens()
                all_tokens = seq_data.get_token_ids_with_draft()
                generation_tokens = all_tokens[num_computed_draft_tokens:]

                draft_lens.append(len(generation_tokens))
                input_tokens.extend(generation_tokens)

                position_start = num_computed_draft_tokens
                for i in range(len(generation_tokens)):
                    position = position_start + i
                    input_positions.append(position)

                for i in range(len(generation_tokens)):
                    context_len = num_computed_draft_tokens + i + 1
                    if self.sliding_window is not None:
                        context_len = min(context_len, self.sliding_window)
                    context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]

                for position in range(position_start, position_start + len(generation_tokens)):
                    block_number = block_table[position // self.block_size]
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping.append(slot)

                    if self.sliding_window is not None:
                        sliding_window_blocks = (self.sliding_window //
                                                 self.block_size)
                        block_table = block_table[-sliding_window_blocks:]
                    block_tables.append(block_table)

        batch_size = len(input_tokens)
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                context_lens.append(1)
                block_tables.append([])
            batch_size = graph_batch_size

        # input_tokens = _async_h2d(
        #     input_tokens, dtype=torch.long, pin_memory=True)
        # input_positions = _async_h2d(
        #     input_positions, dtype=torch.long, pin_memory=True)
        # slot_mapping = _async_h2d(
        #     slot_mapping, dtype=torch.long, pin_memory=True)
        # context_lens = _async_h2d(
        #     context_lens, dtype=torch.int, pin_memory=True)

        input_tokens = torch.tensor(
            input_tokens, dtype=torch.long, device="cuda")
        input_positions = torch.tensor(
            input_positions, dtype=torch.long, device="cuda")
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.long, device="cuda")
        context_lens = torch.tensor(
            context_lens, dtype=torch.int, device="cuda")

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device="cuda")
        else:
            max_block_table_len = max([len(t) for t in block_tables])
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device="cuda",
            )

        input_metadata = InputMetadata(
            num_prefill_tokens=0,
            num_decode_tokens=sum(draft_lens),
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            prefill_lens=None,
            target_lens=None,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            use_target_attention=False,
        )
        return input_tokens, input_positions, input_metadata, draft_lens

    @ nvtx_range("_prepare_sample")
    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        draft_lens: List[int],
        target_lens: List[int],
        draft_probs_tensor: Optional[torch.Tensor] = None,
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0
        sampled_draft_token_ids: List[List[int]] = []
        target_modify_greedy_indices: List[int] = []
        target_modify_greedy_start_idx = 0

        max_prompt_len = max(prompt_lens) if prompt_lens else 1
        prefill_idx = 0
        target_decode_idx = 0
        draft_decode_idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_data = seq_group_metadata.seq_data[seq_ids[0]]
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.spec_decode_stage == SpecDecodeStage.PREFILL:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[prefill_idx]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += prompt_len
                prefill_idx += 1

            elif seq_group_metadata.spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
                num_seqs = len(seq_ids)
                draft_len = draft_lens[draft_decode_idx]
                selected_token_indices.append(
                    selected_token_start_idx + draft_len - 1)
                selected_token_start_idx += draft_len

                categorized_sample_indices[sampling_params.sampling_type].extend(
                    range(categorized_sample_indices_start_idx,
                          categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs
                draft_decode_idx += 1

            elif seq_group_metadata.spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
                assert draft_probs_tensor is not None

                num_seqs = len(seq_ids)
                target_len = target_lens[target_decode_idx]
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + target_len))

                if sampling_params.sampling_type == SamplingType.GREEDY:
                    target_modify_greedy_indices.extend(
                        range(target_modify_greedy_start_idx,
                              target_modify_greedy_start_idx + target_len))

                selected_token_start_idx += target_len
                target_modify_greedy_start_idx += target_len

                # categorized_sample_indices[sampling_params.sampling_type].extend(
                categorized_sample_indices[sampling_params.sampling_type].extend(
                    range(categorized_sample_indices_start_idx,
                          categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

                draft_token_ids = seq_data.get_draft_token_ids()
                assert len(draft_token_ids) == target_len - 1
                sampled_draft_token_ids.append(draft_token_ids)
                target_decode_idx += 1

            else:
                raise ValueError(f"Invalid spec decode stage: "
                                 f"{seq_group_metadata.spec_decode_stage}")

        # selected_token_indices = _async_h2d(selected_token_indices,
        #                                     dtype=torch.long,
        #                                     pin_memory=not self.in_wsl)
        # target_modify_greedy_indices = _async_h2d(target_modify_greedy_indices,
        #                                           dtype=torch.long,
        #                                           pin_memory=not self.in_wsl)
        # categorized_sample_indices = {
        #     t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=not self.in_wsl)
        #     for t, seq_ids in categorized_sample_indices.items()
        # }

        selected_token_indices = torch.tensor(
            selected_token_indices, dtype=torch.long, device="cuda")
        target_modify_greedy_indices = torch.tensor(
            target_modify_greedy_indices, dtype=torch.long, device="cuda")
        categorized_sample_indices = {
            t: torch.tensor(seq_ids, dtype=torch.int, device="cuda")
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        if seq_group_metadata.spec_decode_stage == SpecDecodeStage.TARGET_DECODE:
            sampled_draft_token_ids = _make_tensor_with_pad(sampled_draft_token_ids,
                                                            max_len=max(
                                                                target_lens) - 1,
                                                            pad=0,
                                                            dtype=torch.long)

        else:
            sampled_draft_token_ids = None
            draft_probs_tensor = None

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            draft_lens=draft_lens,
            target_lens=target_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            target_modify_greedy_indices=target_modify_greedy_indices,
            sampled_draft_token_ids=sampled_draft_token_ids,
            draft_probs_tensor=draft_probs_tensor,
            emulate_accept_prob=self.spec_decode_config.emulate_accept_prob,
        )
        return sampling_metadata

    @ nvtx_range("prepare_target_input_tensors")
    def _prepare_target_input_tensors(
        self,
        prefill_seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        target_decode_seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        draft_probs_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata]:

        # Prepare prefill input tensors
        (prefill_input_tokens, prefill_input_positions,
         prefill_slot_mapping, prefill_lens) = self._prepare_prefill(
            prefill_seq_group_metadata_list)

        # Prepare target decode input tensors
        (target_input_tokens, target_input_positions,
         target_slot_mapping, target_context_lens,
         target_block_tables, target_lens) = self._prepare_target_decode(
            target_decode_seq_group_metadata_list)

        sampling_metadata = self._prepare_sample(prefill_seq_group_metadata_list + target_decode_seq_group_metadata_list,
                                                 prefill_lens, [], target_lens, draft_probs_tensor)

        # Combine prefill and target input tokens, input_positions, slot_mappings.
        # input_tokens = _async_h2d(
        #     prefill_input_tokens + target_input_tokens,
        #     dtype=torch.long, pin_memory=True)
        # input_positions = _async_h2d(
        #     prefill_input_positions + target_input_positions,
        #     dtype=torch.long, pin_memory=True)
        # slot_mapping = _async_h2d(
        #     prefill_slot_mapping + target_slot_mapping,
        #     dtype=torch.long, pin_memory=True)

        input_tokens = torch.tensor(
            prefill_input_tokens + target_input_tokens, dtype=torch.long, device="cuda")
        input_positions = torch.tensor(
            prefill_input_positions + target_input_positions, dtype=torch.long, device="cuda")
        slot_mapping = torch.tensor(
            prefill_slot_mapping + target_slot_mapping, dtype=torch.long, device="cuda")

        num_prefill_tokens = sum(prefill_lens)
        num_decode_tokens = sum(target_lens)

        # target_lens = _async_h2d(
        #     target_lens, dtype=torch.int, pin_memory=True)
        target_lens = torch.tensor(
            target_lens, dtype=torch.int, device="cuda")

        if target_decode_seq_group_metadata_list:
            # context_lens = _async_h2d(
            #     target_context_lens, dtype=torch.int, pin_memory=True)
            context_lens = torch.tensor(
                target_context_lens, dtype=torch.int, device="cuda"
            )
            max_block_table_len = max([len(t) for t in target_block_tables])
            block_tables = _make_tensor_with_pad(
                target_block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device="cuda",
            )

            input_metadata = InputMetadata(
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                slot_mapping=slot_mapping,
                max_context_len=max(target_context_lens),
                prefill_lens=prefill_lens,
                target_lens=target_lens,
                context_lens=context_lens,
                block_tables=block_tables,
                use_cuda_graph=False,
                # Target decode can use target attention
                use_target_attention=self.spec_decode_config.target_attention,
            )
        else:
            input_metadata = InputMetadata(
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                slot_mapping=slot_mapping,
                prefill_lens=prefill_lens,
                target_lens=target_lens,
                max_context_len=None,
                context_lens=None,
                block_tables=None,
                use_cuda_graph=False,
                use_target_attention=False,  # Prefill does not use target attention
            )

        return input_tokens, input_positions, input_metadata, sampling_metadata

    @ nvtx_range("prepare_draft_input_tensors")
    def _prepare_draft_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata]:
        # Prepare input tensors.
        (input_tokens, input_positions, input_metadata,
            draft_lens) = self._prepare_draft_decode(
            seq_group_metadata_list)

        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 [], draft_lens, [],
                                                 None)

        return input_tokens, input_positions, input_metadata, sampling_metadata

    @ torch.inference_mode()
    def execute_model(
        self,
        prefill_seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        target_decode_seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        draft_decode_seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_probs_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[SamplerOutput]:
        if self.is_target:
            input_tokens, input_positions, input_metadata, sampling_metadata = (
                self._prepare_target_input_tensors(prefill_seq_group_metadata_list,
                                                   target_decode_seq_group_metadata_list,
                                                   draft_probs_tensor))
        else:
            input_tokens, input_positions, input_metadata, sampling_metadata = (
                self._prepare_draft_input_tensors(draft_decode_seq_group_metadata_list))

        # Execute the model.
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata
        )

        return output

    @ torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
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
                token_chunk_size=seq_len,
                spec_decode_stage=SpecDecodeStage.PREFILL,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers

        input_tokens, input_positions, input_metadata, prompt_len = (
            self._prepare_profile_input(seqs))
        sampling_metadata = self._prepare_sample(
            seqs, prompt_len, [], [], None)

        # Execute the model.
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )

        torch.cuda.synchronize()
        return

    @ torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size,
                                      dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(_BATCH_SIZES_TO_CAPTURE):
            # Create dummy input_metadata.
            input_metadata = InputMetadata(
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                slot_mapping=slot_mapping[:batch_size],
                max_context_len=self.max_context_len_to_capture,
                context_lens=context_lens[:batch_size],
                prefill_lens=None,
                target_lens=None,
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
                use_target_attention=False,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                input_metadata,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
        )
        torch.cuda.synchronize()

        # Capture the graph.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(input_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(input_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(input_metadata.block_tables,
                                                 non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@ nvtx_range("_make_tensor_with_pad")
def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]] = "cuda",
) -> torch.Tensor:
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = np.zeros([len(x), max_len], dtype=np.int32) + pad
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb
    return torch.tensor(padded_x, dtype=dtype, device=device)


def _get_graph_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8


def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)
