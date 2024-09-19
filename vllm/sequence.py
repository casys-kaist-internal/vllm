"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union, Tuple, OrderedDict, Deque
import torch
from collections import deque

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams
from vllm.utils import nvtx_range

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]


class SpecDecodeStage(enum.Enum):
    """The stages of the speculative decoding process."""
    PREFILL = enum.auto()
    DRAFT_DECODE = enum.auto()
    TARGET_DECODE = enum.auto()


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_LENGTH_CAPPED_DRAFT = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

        self.draft_token_ids: List[int] = []
        self.draft_logprobs: List[float] = []

        self._num_computed_target_tokens = 0
        self._num_computed_draft_tokens = 0

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_len_with_draft(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids) + len(self.draft_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_token_ids_with_draft(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids + self.draft_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def get_last_token_ids(self, count: int) -> List[int]:
        if not self.output_token_ids:
            return self.prompt_token_ids[-count:]
        return self.output_token_ids[-count:]

    # Spec Decode
    def append_draft_token_id(self, token_id: int, logprobs: float) -> None:
        self.draft_token_ids.append(token_id)
        self.draft_logprobs.append(logprobs)

    def accept_draft_tokens(self, accept_cnt: int) -> None:
        for i in range(accept_cnt):
            self.append_token_id(self.draft_token_ids[i],
                                 self.draft_logprobs[i])

        self.draft_token_ids.clear()
        self.draft_logprobs.clear()

        self._num_computed_draft_tokens = self.get_len() - 1
        self._num_computed_target_tokens = self.get_len() - 1

    def get_draft_len(self) -> int:
        return len(self.draft_token_ids)

    def get_draft_token_ids(self) -> List[int]:
        return self.draft_token_ids

    def drop_draft_tokens(self, drop_cnt: int) -> None:
        if drop_cnt == 0:
            return
        assert drop_cnt > 0

        # Remove drop_cnt elements from the back of draft_token_ids
        # and draft_logprobs.
        self.draft_token_ids = self.draft_token_ids[:-drop_cnt]
        self.draft_logprobs = self.draft_logprobs[:-drop_cnt]
        self._num_computed_draft_tokens -= drop_cnt

    def get_num_uncomputed_target_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self._num_computed_target_tokens

    def get_num_uncomputed_draft_tokens(self) -> int:
        return self.get_len_with_draft() - self._num_computed_draft_tokens

    def get_num_computed_draft_tokens(self) -> int:
        return self._num_computed_draft_tokens

    def update_num_computed_target_tokens(self, num_computed_tokens):
        self._num_computed_target_tokens += num_computed_tokens

    def update_num_computed_draft_tokens(self, num_computed_tokens):
        self._num_computed_draft_tokens += num_computed_tokens

    def reset_state_for_recompute(self) -> None:
        self._num_computed_target_tokens = 0
        self._num_computed_draft_tokens = 0

        # We clear the draft tokens for reseting state.
        self.draft_token_ids.clear()
        self.draft_logprobs.clear()

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"draft_token_ids={self.draft_token_ids})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    MOVING_AVG_WINDOW_SIZE = 10

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
        draft_size: Optional[int] = 0,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        self.decode_offset = 0

        # Input + output tokens
        self.tokens: Optional[List[str]] = None

        # Spec Decode
        self.draft_size = draft_size
        self.accept_probs: List[float] = []
        self.sampled_draft_probs: List[float] = []
        self.predicted_cumulated_accept_probs: List[float] = []

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_len_with_draft(self) -> int:
        return self.data.get_len_with_draft()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_token_ids_with_draft(self) -> List[int]:
        return self.data.get_token_ids_with_draft()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_last_token_ids(self, count: int) -> List[int]:
        return self.data.get_last_token_ids(count)

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    # Spec Decode
    @nvtx_range("append_draft_token_id")
    def append_draft_token_id(self, token_id: int, logprobs: Dict[int, float],
                              pre_temp_sampled_draft_prob: torch.Tensor) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_draft_token_id(token_id, logprobs[token_id])

        if pre_temp_sampled_draft_prob is not None:
            self.sampled_draft_probs.append(pre_temp_sampled_draft_prob.item())
            assert len(self.sampled_draft_probs) == self.data.get_draft_len()

    @nvtx_range("accept_draft_tokens")
    def accept_draft_tokens(self, accept_cnt: int, accept_prob: List[float]):
        reject_cnt = self.data.get_draft_len() - accept_cnt
        self.data.accept_draft_tokens(accept_cnt)
        self.output_logprobs = self.output_logprobs[:-reject_cnt]
        self._remove_tokens_from_blocks(reject_cnt)

        # if self.predicted_cumulated_accept_probs:
        #     for i in range(len(self.predicted_cumulated_accept_probs)):
        #         accepted = i < accept_cnt
        #         print(
        #             f"result, {accepted}, {self.predicted_cumulated_accept_probs[i]}")

        self.predicted_cumulated_accept_probs.clear()
        self.accept_probs.clear()
        if accept_prob != 0:
            self.accept_probs.extend(accept_prob)

    def get_num_additional_blocks(self, size: int) -> int:
        last_block = self.logical_token_blocks[-1]
        num_empty_slots = last_block.get_num_empty_slots()

        if size <= num_empty_slots:
            return 0

        remaining_slots = size - num_empty_slots
        num_blocks = (remaining_slots + self.block_size - 1) // self.block_size

        return num_blocks

    def _remove_tokens_from_blocks(self, remove_cnt: int) -> None:
        assert len(self.logical_token_blocks) > 0

        for _ in range(remove_cnt):
            last_block = self.logical_token_blocks[-1]
            last_block.remove_token()

            if last_block.is_empty():
                self.logical_token_blocks.pop()

    def drop_draft_tokens(self, drop_cnt: int) -> None:
        self.data.drop_draft_tokens(drop_cnt)
        self._remove_tokens_from_blocks(drop_cnt)

        total_length = self.data.get_len_with_draft()
        logical_token_blocks_len = len(self.logical_token_blocks)
        logical_length = (logical_token_blocks_len - 1) * self.logical_token_blocks[0].block_size + self.logical_token_blocks[-1].num_tokens
        assert total_length == logical_length


    def get_draft_len(self) -> int:
        return self.data.get_draft_len()

    def get_num_uncomputed_target_tokens(self) -> int:
        return self.data.get_num_uncomputed_target_tokens()

    def get_num_uncomputed_draft_tokens(self) -> int:
        return self.data.get_num_uncomputed_draft_tokens()

    def get_num_computed_draft_tokens(self) -> int:
        return self.data.get_num_computed_draft_tokens()

    def update_num_computed_target_tokens(self,
                                          num_computed_tokens: int) -> None:
        self.data.update_num_computed_target_tokens(num_computed_tokens)

    def update_num_computed_draft_tokens(self,
                                         num_computed_tokens: int) -> None:
        self.data.update_num_computed_draft_tokens(num_computed_tokens)

    def reset_state_for_recompute(self) -> None:
        self.data.reset_state_for_recompute()
        self.sampled_draft_probs.clear()
        self.accept_probs.clear()
        self.predicted_cumulated_accept_probs.clear()

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.spec_decode_stage = None

    @ property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @ property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.best_of > self.num_seqs():
                # At prompt stage, the sequence group is not yet filled up
                # and only have one sequence running. However, in the
                # generation stage, we will have `best_of` sequences running.
                return self.sampling_params.best_of
            # At sampling stages, return the number of actual sequences
            # that are not finished yet.
            return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.status == status
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)}), "
                f"seqs={self.seqs_dict})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: OrderedDict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        token_chunk_size: Optional[int] = 0,
        spec_decode_stage: Optional[SpecDecodeStage] = None,
        draft_size: Optional[int] = 0,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.token_chunk_size = token_chunk_size
        self.spec_decode_stage = spec_decode_stage
        self.draft_size = draft_size


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
        draft_probs: Optional[torch.Tensor] = None,
        pre_temp_sampled_draft_prob: Optional[float] = 0.0,
        accept_cnt: Optional[int] = 0,
        accept_prob: Optional[torch.Tensor] = 0.0,
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs
        self.draft_probs = draft_probs
        self.pre_temp_sampled_draft_prob = pre_temp_sampled_draft_prob
        self.accept_cnt = accept_cnt
        self.accept_prob = accept_prob

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


class SequenceGroupOutput:
    """The model output associated with a sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"SequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


# For each sequence group, we generate a list of SequenceOutput object,
# each of which contains one possible candidate for the next token.
SamplerOutput = Tuple[List[SequenceGroupOutput], torch.Tensor]
