"""Sequence and its related classes."""
import copy
import enum
from typing import Dict, List, Optional, Union

import torch
from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams

import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]


class SpSStage(enum.Enum):
    """The stage of the speculative sampling process"""
    PROMPT = enum.auto()
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

    # SpS related status
    SPS_ALL_ACCEPT = enum.auto()

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
        # number of tokens that cached with draft KV
        self.draft_cache_cnt = 0
        self.draft_token_ids: List[int] = []
        self.draft_logprobs: List[float] = []
        self.draft_probs: List[torch.Tensor] = []

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    # SpS related methods start
    def get_last_nth_token_id(self, idx) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[idx]
        return self.output_token_ids[idx]

    def get_last_draft_token_id(self) -> int:
        if not self.draft_token_ids:
            return self.get_last_token_id()
        return self.draft_token_ids[-1]

    def get_uncached_draft_token_ids(self) -> List[int]:
        all_tokens_including_draft = self.get_token_ids() + self.get_draft_token_ids()
        uncached_draft_token_ids = all_tokens_including_draft[self.draft_cache_cnt:]

        # NOTE: draft_cache_cnt is updated here
        self.draft_cache_cnt += len(uncached_draft_token_ids)

        return uncached_draft_token_ids

    def get_uncached_draft_len(self) -> int:
        all_tokens_including_draft_cnt = len(self.get_token_ids()) + len(self.get_draft_token_ids())
        uncached_draft_len = all_tokens_including_draft_cnt - self.draft_cache_cnt

        return uncached_draft_len

    def get_draft_cache_cnt(self) -> int:
        return self.draft_cache_cnt

    def append_draft_token_id(self, token_id: int, logprobs: float, probs: torch.Tensor) -> None:
        self.draft_token_ids.append(token_id)
        self.draft_logprobs.append(logprobs)
        self.draft_probs.append(probs)

    def accept_draft_tokens(self, accept_cnt: int) -> None:
        for i in range(accept_cnt):
            self.append_token_id(
                self.draft_token_ids[i], self.draft_logprobs[i])
        
        print(self.draft_token_ids)
        self.draft_cache_cnt = self.get_len() - 1

        self.draft_token_ids.clear()
        self.draft_logprobs.clear()
        self.draft_probs.clear()

    def get_draft_len(self) -> int:
        return len(self.draft_token_ids)

    def get_draft_token_ids(self) -> List[int]:
        return self.draft_token_ids

    def get_draft_probs(self) -> List[torch.Tensor]:
        return self.draft_probs

    def get_draft_prob_for_tokens(self) -> List[float]:
        result = []
        for i, draft_token_id in enumerate(self.draft_token_ids):
            result.append(self.draft_probs[i][draft_token_id].item())

        return result

    def get_last_draft_token_id(self) -> int:
        if len(self.draft_token_ids) == 0:
            return self.get_last_token_id()

        return self.draft_token_ids[-1]
    # SpS related methods end

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}), "
                f"draft_token_ids={self.draft_token_ids}, "
                f"draft_cumulative_logprobs={self.draft_logprobs}")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

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

        # SpS related params
        self.accept_cnt_list: List[int] = []
        self.reject_pos: List[int] = []
        self.accept_probs: List[float] = []
        self.beta_list: List[float] = []
        self.last_ema = None  # This will store the last calculated EMA value
        self.last_calculated_index = -1  # Tracks the last index for which EMA was calculated
        self.cumulative_accept_prob = 1

        self.bonus_token_id = None
        self.bonus_logprobs = None

        self.correlation_x = []
        self.correlation_y = []
        self.correlation_z = []

        # Can change every iteration if use_dynamic_draft_size is True
        self.draft_size = draft_size

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

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

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

    # SpS related methods start
    def save_bonus_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        assert self.bonus_token_id is None and self.bonus_logprobs is None
        
        self.bonus_token_id = token_id
        self.bonus_logprobs = logprobs

    def append_bonus_token_id(self) -> None:
        assert self.bonus_token_id is not None and self.bonus_logprobs is not None

        self.append_token_id(self.bonus_token_id,
                             self.bonus_logprobs)
        
        self.bonus_token_id = None
        self.bonus_logprobs = None

    def _remove_tokens_from_blocks(self, remove_cnt: int) -> None:
        assert len(self.logical_token_blocks) > 0
        free_block_cnt = 0

        for _ in range(remove_cnt):
            last_block = self.logical_token_blocks[-1]
            last_block.remove_token()

            if last_block.is_empty():
                self.logical_token_blocks.pop()
                free_block_cnt += 1

        return free_block_cnt

    def get_draft_token_ids(self) -> List[int]:
        return self.data.get_draft_token_ids()

    def get_draft_probs(self) -> List[torch.Tensor]:
        return self.data.get_draft_probs()

    def get_draft_len(self) -> int:
        return self.data.get_draft_len()
    
    # Calculates the Exponential Moving Average (EMA) of beta values
    def get_beta_ema(self) -> float:
        # Ensure there is at least one beta to calculate EMA
        if len(self.beta_list) < 3:
            return 0.5  # Return a default initial beta value if list is empty

        # Define the span for EMA calculation
        decay = 0.5

        # Initialize EMA; if no previous EMAs, start with the first beta value
        if self.last_ema is None:
            self.last_ema = self.beta_list[0]

         # Update EMA only for new beta values added since last calculation
        for beta in self.beta_list[self.last_calculated_index+1:]:
            self.last_ema = decay * beta + (1 - decay) * self.last_ema

        # Update the last calculated index
        self.last_calculated_index = len(self.beta_list) - 1

        return self.last_ema
        # if self.last_ema == 1:
        #     # Avoid division by Zero 
        #     self.last_ema = 0.9999999999999999
    
        # # This is for E(# of expected tokens)
        # return (1 - self.last_ema**(7 + 1)) / (1 - self.last_ema)
        
    def append_draft_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
        probs: torch.Tensor
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_draft_token_id(token_id, logprobs[token_id], probs)

    def check_early_stop(self) -> bool:
        # Check if the sequence should be stopped early
        # Get probability of last draft token
        last_draft_token_id = self.data.get_last_draft_token_id()
        draft_prob = self.data.get_draft_probs()[-1][last_draft_token_id].item()
        beta_ema = self.get_beta_ema()
        # predicted_accept_prob = 0.48*beta_ema + 1.07*draft_prob + -0.05*beta_ema**2 + -0.46*beta_ema*draft_prob + -1.98*draft_prob**2 + -0.16*beta_ema**3 + 0.35*(beta_ema**2)*draft_prob + -0.10*beta_ema*draft_prob**2 + 1.62*draft_prob**3 + 0.22
        # predicted_accept_prob = 0.1*beta_ema + 0.9*draft_prob 
        # predicted_accept_prob = 0.59*beta_ema + 1.00*draft_prob + -0.51*beta_ema**2 + -0.25*beta_ema*draft_prob + -2.04*draft_prob**2 + 0.18*beta_ema**3 + 0.19*(beta_ema**2)*draft_prob + -0.07*beta_ema*draft_prob**2 + 1.65*draft_prob**3 + 0.23
        # 0.33*beta_ema + 0.73*draft_prob + -0.28*beta_ema^2 + 0.03*beta_ema draft_prob + -1.33*draft_prob^2 + 0.09*beta_ema^3 + 0.05*beta_ema^2 draft_prob + -0.15*beta_ema draft_prob^2 + 1.16*draft_prob^3 + 0.32
        # predicted_accept_prob = 0.33*beta_ema + 0.73*draft_prob + -0.28*beta_ema**2 + 0.03*beta_ema*draft_prob + -1.33*draft_prob**2 + 0.09*beta_ema**3 + 0.05*(beta_ema**2)*draft_prob + -0.15*beta_ema*(draft_prob**2) + 1.16*draft_prob**3 + 0.32
        # 0.97*beta_ema_binned + 1.13*draft_prob_binned + -1.22*beta_ema_binned^2 + -0.08*beta_ema_binned draft_prob_binned + -1.03*draft_prob_binned^2 + 0.59*beta_ema_binned^3 + 0.08*beta_ema_binned^2 draft_prob_binned + -0.26*beta_ema_binned draft_prob_binned^2 + 0.82*draft_prob_binned^3 + 0.08
        predicted_accept_prob = 0.97*beta_ema + 1.13*draft_prob + -1.22*beta_ema**2 + -0.08*beta_ema*draft_prob + -1.03*draft_prob**2 + 0.59*beta_ema**3 + 0.08*(beta_ema**2)*draft_prob + -0.26*beta_ema*(draft_prob**2) + 0.82*draft_prob**3 + 0.08
        # print(predicted_accept_prob, beta_ema, draft_prob)

        self.cumulative_accept_prob *= predicted_accept_prob

        random_accept_prob = np.random.uniform(0, 1)
        if self.cumulative_accept_prob < random_accept_prob:
            self.cumulative_accept_prob = 1
            return True
        else:
            return False

        return (predicted_accept_prob < 0.5)

    def custom_score(self, y_true, y_pred):
        # Convert predictions to nearest integers
        nearest_int_pred = np.round(y_pred)
        
        # Calculate the absolute difference
        abs_errors = np.abs(y_true - nearest_int_pred)
        
        # Mean Absolute Error based on integer predictions
        mae = np.mean(abs_errors)
        return mae

    def accept_draft_tokens(self,
                            accept_cnt: int,
                            accept_probs: List[float],
                            beta_list: List[float]) -> int:
        # assert accept_cnt <= self.draft_size
        assert self.draft_size == self.get_draft_len()
        reject_cnt = self.draft_size - accept_cnt
        # print("accept ", " | ",  accept_cnt, " | ", self.beta_list,  " | ", self.data.get_draft_prob_for_tokens(),  " | ", self.accept_cnt_list, " | ", accept_probs,  " | ", beta_list)

        self.data.accept_draft_tokens(accept_cnt)
        self.output_logprobs = self.output_logprobs[:-reject_cnt]
        print(accept_cnt, self.draft_size)
        # We overprovisioned the blocks when scheduling considering the draft size + bonus token 
        # Need to free the blocks that are not used
        # If all tokens are accepted (reject_cnt equals 0), we don't need to free any blocks
        free_block_cnt = self._remove_tokens_from_blocks(reject_cnt)
        # self.correlation_x.append(accept_cnt)
        # ema = self.get_beta_ema()
        # self.correlation_y.append(self.draft_size)
        # print(self.custom_score(np.array(self.correlation_x), np.array(self.correlation_y)))

        # if accept_cnt != self.draft_size:
        #     accept_probs = accept_probs[:accept_cnt+1]
        #     beta_list = beta_list[:accept_cnt+1]

        # else:  # all accept bonus token
        #     accept_probs.append(1)
        #     beta_list.append(1)

        # print("!", self.get_beta_ema(), accept_cnt)

        self.accept_cnt_list.append(accept_cnt)
        self.accept_probs.extend(accept_probs)
        self.beta_list.extend(beta_list)

        return free_block_cnt

    def get_last_nth_token_id(self, idx) -> int:
        return self.data.get_last_nth_token_id(idx)

    def get_num_additional_blocks(self, size: int) -> int:
        last_block = self.logical_token_blocks[-1]
        num_empty_slots = last_block.get_num_empty_slots()

        if size <= num_empty_slots:
            return 0

        num_blocks = 1
        remaining_slots = size - num_empty_slots
        num_blocks += (remaining_slots + self.block_size -
                       1) // self.block_size

        return num_blocks

    # SpS related methods end

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

    @property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
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
                f"num_seqs={len(self.seqs_dict)})")


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
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
        sps_stage: Optional[SpSStage] = None,
        draft_size: Optional[int] = 0,
        seq_group: Optional[SequenceGroup] = None,
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables
        self.sps_stage = sps_stage
        self.draft_size = draft_size
        self.seq_group = seq_group


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
        probs: Optional[torch.Tensor] = None,
        total_cnt: Optional[int] = 0,
        accept_cnt: Optional[int] = 0,
        accept_probs: Optional[List[float]] = None,
        beta_list: Optional[List[float]] = None,
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

        # SpS related start
        self.probs = probs
        self.total_cnt = total_cnt
        self.accept_cnt = accept_cnt
        self.accept_probs = accept_probs
        self.beta_list = beta_list
        # SpS related end

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
SamplerOutput = List[SequenceGroupOutput]
