from typing import List, Optional

import torch


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
    """

    def __init__(
        self,
        prompt_lens: List[int],
        draft_lens: List[int],
        target_lens: List[int],
        slot_mapping: torch.Tensor,
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> None:
        self.prompt_lens = prompt_lens
        self.target_lens = target_lens
        self.draft_lens = draft_lens
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables

        # SpS PROMPT: is_prompt == True
        # SpS DRAFT_DECODE: is_prompt == False and is_target_decode == False
        # SpS TARGET_DECODE: is_prompt == False and is_target_decode == True
        self.is_prompt = len(prompt_lens) > 0
        self.is_target_decode = (not self.is_prompt) and (len(target_lens) > 0)

        if self.is_prompt:
            assert self.is_target_decode == False

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"prompt_lens={self.prompt_lens}, "
                f"draft_lens={self.draft_lens}, "
                f"max_context_len={self.max_context_len}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables})")
