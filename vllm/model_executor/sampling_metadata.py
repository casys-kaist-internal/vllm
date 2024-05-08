from typing import Dict, List, Tuple, Optional

import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData


class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        draft_lens: Lengths of drafts.
        selected_token_indices: Token indices selected for sampling.
        categorized_sample_indices: SamplingType -> token indicies to sample.
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        draft_lens: List[int],
        target_lens: List[int],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Dict[SamplingType, torch.Tensor],
        sampled_draft_token_ids: Optional[torch.Tensor] = None,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.draft_lens = draft_lens
        self.target_lens = target_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.num_prompts = len(prompt_lens)
        self.is_prompt = len(prompt_lens) > 0

        # SpS related attributes start
        self.sampled_draft_token_ids = sampled_draft_token_ids
        self.draft_probs = draft_probs
        self.is_draft_decode = len(draft_lens) > 0
        self.is_target_decode = len(target_lens) > 0
        # SpS related attributes end

        # if self.is_target_decode:
        #     assert self.sampled_draft_token_ids is not None
        #     assert self.draft_probs is not None

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
            f"seq_data={self.seq_data}, "
            f"prompt_lens={self.prompt_lens}, "
            f"draft_lens={self.draft_lens}, "
            f"target_lens={self.target_lens}, "
            f"selected_token_indices={self.selected_token_indices}, "
            f"categorized_sample_indices={self.categorized_sample_indices})")
