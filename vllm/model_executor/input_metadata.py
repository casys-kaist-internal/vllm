from typing import Optional, List

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
        num_prefill_tokens: int,
        num_chunked_prefill_tokens: int,
        num_decode_tokens: int,
        slot_mapping: torch.Tensor,
        max_context_len: Optional[int],
        prefill_lens: Optional[List[int]],
        chunked_prefill_lens: Optional[torch.Tensor],
        query_start_locs: Optional[torch.Tensor],
        target_lens: Optional[torch.Tensor],
        chunked_context_lens: Optional[torch.Tensor],
        context_lens: Optional[torch.Tensor],
        chunked_block_tables: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        use_gamma_mapping_attention: bool,
    ) -> None:
        self.num_prefill_tokens = num_prefill_tokens
        self.num_chunked_prefill_tokens = num_chunked_prefill_tokens
        self.num_decode_tokens = num_decode_tokens
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.prefill_lens = prefill_lens
        self.chunked_prefill_lens = chunked_prefill_lens
        self.query_start_locs = query_start_locs
        self.target_lens = target_lens
        self.chunked_context_lens = chunked_context_lens
        self.context_lens = context_lens
        self.chunked_block_tables = chunked_block_tables
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph
        self.use_gamma_mapping_attention = use_gamma_mapping_attention

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return ("InputMetadata("
                f"max_context_len={self.max_context_len}, "
                f"slot_mapping={self.slot_mapping}, "
                f"context_lens={self.context_lens}, "
                f"block_tables={self.block_tables}, "
                f"use_cuda_graph={self.use_cuda_graph})")
