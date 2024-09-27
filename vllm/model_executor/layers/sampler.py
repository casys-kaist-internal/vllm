"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_gather)
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors, MinimizedSamplingTensors
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SamplerOutput,
                           SequenceData, SequenceGroupOutput, SequenceOutput, SpecDecodeStage)
from vllm.utils import nvtx_range


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence, frequency and repetition penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    @nvtx_range("sampler forward")
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[SamplerOutput]:

        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(
            hidden_states, sampling_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)

        # Only perform sampling in the driver worker.
        # Note: `_get_logits` is still distributed across TP workers because
        # the `embedding` weight is distributed across TP workers.
        # TODO(zhuohan): Change the get_logits part to a separate stage.
        if not sampling_metadata.perform_sampling:
            return None

        assert logits is not None
        _, vocab_size = logits.shape

        # Apply logits processors (if any).
        logits = _apply_logits_processors(logits, sampling_metadata)

        torch.cuda.nvtx.range_push("from_sampling_metadata")
        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_penalties, do_top_p_top_k,
            do_min_p) = SamplingTensors.from_sampling_metadata(
            sampling_metadata, vocab_size, logits.device, logits.dtype)
        torch.cuda.nvtx.range_pop()

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)
        pre_temp_probs = None
        if not sampling_metadata.is_target:
            # If draft sampler, we need the probability of the sampled draft token
            # before applying the temperature
            pre_temp_probs = torch.softmax(
                logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_p_top_k(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        if (not sampling_metadata.is_target or len(sampling_metadata.target_lens) == 0 or
                all([l == 1 for l in sampling_metadata.target_lens])):
            sample_results = _sample(probs, logprobs, sampling_metadata)
            accept_cnts = None
            accept_probs = None

        else:
            sample_results, accept_cnts, accept_probs, logprobs = _spec_decode_sample(
                probs, sampling_metadata)

        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)

        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs,
                                     probs, pre_temp_probs,
                                     accept_cnts, accept_probs)


@nvtx_range("_get_logits")
def _get_logits(hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor],
                vocab_size: int) -> Optional[torch.Tensor]:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :vocab_size]
    return logits


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    return hidden_states.index_select(0,
                                      sampling_metadata.selected_token_indices)


def _get_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    logits_row_idx = 0
    found_logits_processors = False
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True
            for seq_id in seq_ids:
                logits_row = logits[logits_row_idx]
                token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
                for logits_processor in logits_processors:
                    logits_row = logits_processor(token_ids, logits_row)
                logits[logits_row_idx] = logits_row
                logits_row_idx += 1
        else:
            logits_row_idx += len(seq_ids)
    if found_logits_processors:
        assert logits_row_idx == logits.shape[0]
    return logits


@nvtx_range("_apply_penalties")
def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size,
                                              num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
    repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties,
                         logits * repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits


def _apply_top_p_top_k(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1).sub_(probs_sort)
    top_p_mask = probs_sum > p.unsqueeze_(dim=1)

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze_(dim=1)

    # Final mask.
    mask = (top_p_mask | top_k_mask)
    logits_sort.masked_fill_(mask, -float("inf"))

    # Re-sort the probabilities.
    src = torch.arange(logits_idx.shape[-1],
                       device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                           index=logits_idx,
                                                           src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits


def _apply_min_p(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """
    Adapted from
    https://github.com/oobabooga/text-generation-webui/blob/3146124ec01f02c8fb1650a6517cf1b60b537aaf/modules/sampler_hijack.py#L16C17-L16C17
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits


def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    samples = samples.tolist()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, _ = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx]]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    random_samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # Find the maximum best_of value of the prompt phase requests.
    random_samples = random_samples.cpu()
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = random_samples[
                sample_idx, :sampling_params.best_of].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _beam_search_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    seq_data: Dict[int, SequenceData],
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # We sample 2 * beam_width candidates to make sure that with high
    # probability we can get `beam_width` candidates in addition to
    # the finished sequences for the next iteration. See
    # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
    # for details. See also HF reference:
    # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
    #
    # NOTE: Beam search is not vectorized, so its speed can be slower than
    # other sampling methods.
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        beam_width = sampling_params.best_of
        seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * (2 * beam_width)
            _, next_token_ids = torch.topk(seq_group_logprobs[0],
                                           2 * beam_width)
            next_token_ids = next_token_ids.tolist()
        else:
            # Generation phase.
            cumulative_logprobs = [
                seq_data[seq_id].cumulative_logprob for seq_id in seq_ids
            ]
            cumulative_logprobs = torch.tensor(
                cumulative_logprobs,
                dtype=torch.float,
                device=seq_group_logprobs.device)
            seq_group_logprobs = (seq_group_logprobs +
                                  cumulative_logprobs.unsqueeze(dim=1))
            _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
                                     2 * beam_width)
            topk_ids = topk_ids.tolist()
            vocab_size = seq_group_logprobs.size(-1)
            parent_ids = [i // vocab_size for i in topk_ids]
            next_token_ids = [i % vocab_size for i in topk_ids]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
@nvtx_range("_multinomial")
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
):
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs).exponential_(1)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


@nvtx_range("_sample")
def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> List[Tuple[List[int], List[int]]]:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}

    # Counterintiutively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type in SamplingType:
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices)
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[sample_indices], dim=-1)
            # Needed for spec decoding greedy verification test.
            _modify_greedy_probs_inplace(probs,
                                         sample_indices,
                                         greedy_samples)
        elif sampling_type == SamplingType.RANDOM:
            max_best_of = 1
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of = max(max_best_of, sampling_params.best_of)
            multinomial_samples = _multinomial(probs[sample_indices],
                                               max_best_of)
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # GPU<->CPU sync happens in the loop below.

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        seq_group_ids, seq_groups, is_prompts, sample_indices = sample_metadata[
            sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type == SamplingType.RANDOM:
            sample_results = _random_sample(seq_groups, is_prompts,
                                            multinomial_samples)
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results


@nvtx_range("_spec_decode_sample")
def _spec_decode_sample(
    probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> Tuple[List[Tuple[List[int], List[int]]], torch.Tensor, torch.Tensor]:
    # Divide probs into prefill and decode probs
    prefill_probs = probs[:sampling_metadata.num_prompts]
    decode_probs = probs[sampling_metadata.num_prompts:]

    # Needed for spec decoding greedy verification test.
    # check empty tensor
    if sampling_metadata.target_modify_greedy_indices is not None:
        greedy_samples = torch.argmax(
            decode_probs[sampling_metadata.target_modify_greedy_indices], dim=-1)
        _modify_greedy_probs_inplace(decode_probs,
                                     sampling_metadata.target_modify_greedy_indices,
                                     greedy_samples)
        del greedy_samples

    # Target lens includes one additional token just before draft tokens and draft tokens.
    target_lens = sampling_metadata.target_lens
    target_lens_minus_one = [l - 1 for l in target_lens]

    # sampled_draft_token_ids: [seq_idx, target_lens_minus_one]
    sampled_draft_token_ids = sampling_metadata.sampled_draft_token_ids

    # target_probs: [seq_len, vocab_size] -> [seq_idx, target_lens, vocab_size]
    target_probs = _reshape_and_pad(
        decode_probs, target_lens, decode_probs.size(-1))

    # draft_probs: [seq_len, vocab_size] -> [seq_idx, target_lens_minus_one, vocab_size]
    draft_probs = _reshape_and_pad(
        sampling_metadata.draft_probs_tensor,
        target_lens_minus_one, target_probs.size(-1)
    )

    # target_probs_for_sampled_draft_token: [seq_idx, target_lens_minus_one]
    target_prob_for_sampled_draft_token = torch.gather(
        target_probs, 2, sampled_draft_token_ids.unsqueeze(-1)).squeeze(-1)

    # draft_probs_for_sampled_draft_token: [seq_idx, target_lens_minus_one]
    draft_prob_for_sampled_draft_token = torch.gather(
        draft_probs, 2, sampled_draft_token_ids.unsqueeze(-1)).squeeze(-1)

    # print("target_probs_for_sampled_draft_token",
    #       target_prob_for_sampled_draft_token)
    # print("draft_probs_for_sampled_draft_token",
    #       draft_prob_for_sampled_draft_token)

    # accept_probs: [seq_idx, target_lens_minus_one]
    accept_probs = target_prob_for_sampled_draft_token.div_(
        draft_prob_for_sampled_draft_token)

    del probs, target_prob_for_sampled_draft_token, draft_prob_for_sampled_draft_token

    # Replace inf and nan values with 0
    accept_probs[torch.isinf(accept_probs) | torch.isnan(accept_probs)] = 0

    # Clamp the values to the range [0, 1]
    accept_probs = torch.clamp(accept_probs, max=1)

    random_prob = torch.rand_like(accept_probs)

    # accept is 0 and reject is 1
    accepted = torch.where(
        random_prob < accept_probs,
        torch.zeros_like(accept_probs), torch.ones_like(accept_probs))

    # cumulative sum
    accepted.cumsum_(dim=1)

    # create a mask that contains 1 until the first reject
    accepted = (accepted == 0)

    # accept_cnts: [seq_idx]
    accept_cnts = torch.sum(accepted, dim=1)

    del accepted

    # Cap accept_cnts to not exceed target_lens_minus_one
    target_lens_tensor = torch.tensor(
        target_lens_minus_one, device='cuda')
    accept_cnts = torch.min(accept_cnts, target_lens_tensor)
    all_accept_mask = (accept_cnts == target_lens_tensor)

    # make accept_cnt 0 for all accepted sequences.
    masked_accept_cnt = accept_cnts.clone()
    masked_accept_cnt[all_accept_mask] = 0

    # target_prob_at_rejected_draft_idx: [seq_idx, vocab_size]
    # draft_prob_at_rejected_draft_idx: [seq_idx, vocab_size]
    indices = torch.arange(target_probs.size(0), device='cuda')
    target_prob_at_reject_idx = target_probs[indices,
                                             masked_accept_cnt, :].squeeze(1)

    draft_prob_at_reject_idx = draft_probs[indices, masked_accept_cnt, :].squeeze(
        1)
    del masked_accept_cnt

    # modified_rejection_prob: [seq_idx, vocab_size]
    modified_rejection_prob = _get_modified_rejection_prob(
        target_prob_at_reject_idx, draft_prob_at_reject_idx)
    del target_prob_at_reject_idx, draft_prob_at_reject_idx

    # recover to original probability for all accepted sequences
    modified_rejection_prob[all_accept_mask, :] = target_probs[indices,
                                                               target_lens_tensor, :][all_accept_mask].squeeze(1)

    # concat prefill_probs and modified_rejection_prob
    modified_rejection_prob = torch.cat(
        [prefill_probs, modified_rejection_prob], dim=0)

    modified_rejection_logprobs = torch.log(modified_rejection_prob)
    del target_probs, draft_probs, target_lens_tensor, indices

    sample_results = _sample(modified_rejection_prob,
                             modified_rejection_logprobs,
                             sampling_metadata)

    return sample_results, accept_cnts, accept_probs, modified_rejection_logprobs


@ nvtx_range("_get_logprobs")
def _get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[List[Optional[Dict[int, float]]]]], List[List[Dict[
        int, float]]]]:
    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
    sample_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        num_parent_seqs = len(seq_ids)
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            batched_logprobs_query_seq_indices.extend(
                sample_idx + j for j in range(prompt_len - 1))
            batched_logprobs_query_token_indices.extend(
                token_id for token_id in prompt_tokens[1:])
            sample_idx += prompt_len - 1
        batched_logprobs_query_seq_indices.extend(
            [sample_idx + parent_id for parent_id in parent_ids])
        batched_logprobs_query_token_indices.extend(next_token_ids)
        if sampling_params.logprobs is not None:
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.logprobs)
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)

    # Batched query for logprobs of selected token
    batched_logprobs_query_result = logprobs[[
        batched_logprobs_query_seq_indices,
        batched_logprobs_query_token_indices
    ]]

    # Batched query for logprobs of topk tokens
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(logprobs,
                                                 largest_num_logprobs,
                                                 dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = None, None

    batched_logprobs_query_result = batched_logprobs_query_result.cpu()

    # Gather results
    result_prompt_logprobs: List[Optional[PromptLogprobs]] = []
    result_sample_logprobs: List[SampleLogprobs] = []
    sample_idx = 0
    query_result_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result

        # Prompt logprobs
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            num_logprobs = sampling_params.prompt_logprobs
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            group_prompt_logprobs: PromptLogprobs = [None]
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {
                    token_id:
                    batched_logprobs_query_result[query_result_idx].item()
                }
                if num_logprobs > 0:
                    prompt_logprobs_dict.update(
                        zip(top_token_ids[sample_idx, :num_logprobs].tolist(),
                            top_logprobs[sample_idx, :num_logprobs].tolist()))
                group_prompt_logprobs.append(prompt_logprobs_dict)
                sample_idx += 1
                query_result_idx += 1
            result_prompt_logprobs.append(group_prompt_logprobs)
        else:
            result_prompt_logprobs.append(None)

        # Sample logprobs
        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SampleLogprobs = []
        for next_token_id, parent_id in zip(next_token_ids, parent_ids):
            sample_logprobs_dict = {
                next_token_id:
                batched_logprobs_query_result[query_result_idx].item()
            }
            query_result_idx += 1
            if num_logprobs > 0:
                sample_logprobs_dict.update(
                    zip(
                        top_token_ids[sample_idx +
                                      parent_id, :num_logprobs].tolist(),
                        top_logprobs[sample_idx +
                                     parent_id, :num_logprobs].tolist()))
            group_sample_logprobs.append(sample_logprobs_dict)
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)

    return result_prompt_logprobs, result_sample_logprobs


@ nvtx_range("_build_sampler_output")
def _build_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
    draft_probs: Optional[torch.Tensor] = None,
    pre_temp_draft_probs: Optional[torch.Tensor] = None,
    accept_cnts: Optional[torch.Tensor] = None,
    accept_probs: Optional[torch.Tensor] = None,
) -> SamplerOutput:
    sampler_output = []

    # GPU -> CPU sync
    accept_cnts = accept_cnts.tolist() if accept_cnts is not None else None
    accept_probs = accept_probs.tolist() if accept_probs is not None else None

    for (idx, (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs)) in enumerate(zip(sampling_metadata.seq_groups,
                                                  sample_results, prompt_logprobs,
                                                  sample_logprobs)):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids,
                                                      next_token_ids,
                                                      group_sample_logprobs):
            if sampling_metadata.is_target:
                if idx < sampling_metadata.num_prompts:
                    seq_outputs.append(
                        SequenceOutput(seq_ids[parent_id], next_token_id, logprobs))
                else:
                    decode_idx = idx - sampling_metadata.num_prompts
                    accept_cnt = accept_cnts[decode_idx] if accept_cnts is not None else 0
                    accept_prob = accept_probs[decode_idx] if accept_probs is not None else 0
                    seq_outputs.append(
                        SequenceOutput(seq_ids[parent_id], next_token_id, logprobs,
                                       accept_cnt=accept_cnt, accept_prob=accept_prob))
            else:
                if sampling_metadata.selective_validation:
                    pre_temp_sampled_draft_prob = pre_temp_draft_probs[
                        idx][next_token_id]
                else:
                    pre_temp_sampled_draft_prob = None

                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id, logprobs,
                                   draft_probs=draft_probs[idx],
                                   pre_temp_sampled_draft_prob=pre_temp_sampled_draft_prob))

        sampler_output.append(
            SequenceGroupOutput(seq_outputs, group_prompt_logprobs))

    del pre_temp_draft_probs

    return sampler_output


@nvtx_range("_reshape_and_pad")
def _reshape_and_pad(
        x: torch.Tensor,
        lens: List[int],
        vocab_size: int
):
    max_size = max(lens)
    padded_x = torch.zeros(
        (len(lens), max_size, vocab_size), device=x.device)

    idx = 0
    for i, size in enumerate(lens):
        padded_x[i, :size, :x.size(-1)] = x[idx:idx + size]
        idx += size

    return padded_x


def _get_modified_rejection_prob(target_prob: torch.Tensor, draft_prob: torch.Tensor) -> torch.Tensor:
    # min_val = _smallest_positive_value(target_prob.dtype)
    target_prob.sub_(draft_prob)
    target_prob.clamp_(min=0)
    x_max_sum = target_prob.sum(dim=-1, keepdim=True)
    target_prob.div_(x_max_sum)
    return target_prob


def _smallest_positive_value(dtype) -> float:
    """Return the smallest positive normal value representable by the dtype."""
    return torch.finfo(dtype).tiny


def _modify_greedy_probs_inplace(probs: torch.Tensor,
                                 sample_indices: torch.Tensor,
                                 greedy_samples: torch.Tensor) -> None:
    """Modify the probability distributions of the greedily-sampled tokens such
        that each sampled token has a "probability" of 1.0. This is required by
        speculative decoding, which depends on the sampling method being encoded
        within the probability distribution for correctness.

        # Why do we only need to do this for greedy sampling?

        vLLM's sampler performs the following steps for greedy or multinomial
        (random) sampling:
            1. Get logits from model.
            2. Modify logits according to per-sequence sampling parameters.
                - Multiply by temperature, top-k and top-p masking, penalize tokens
                    according to their frequency, etc.
            3. Sample a token.
                - Random sampling simply samples from the modified probability
                    distribution.
                - Greedy sampling performs `argmax` to obtain the token with the
                    highest likelihood.

        Ignoring greedy sampling for a moment, we find that the computed probability
        distribution has the following property: we can sample from it independently
        and find that the token sampled by the Sampler has a frequency corresponding
        to how often we see it in our sampling. In other words, for tokens sampled
        with vLLM's random SamplingType, the computed probability distribution
        encodes the sampling methodology completely.

        Greedy sampling does not normally have this property. vLLM modifies logits
        according to sampling params, then performs `argmax`, then returns the
        sampled token and the computed probability distribution. If we sample from
        the distribution, we'll find the likelihood of the greedily-sampled token
        is not always 1.0.

        Since lossless speculative decoding requires that the sampling methodology
        be encoded within the probability distribution, we are motivated to modify
        the probability distribution such that the sampled token has probability 1
        when speculative decoding is used.

        NOTE: Alternatively, we could use an extremely low temperature to achieve
        greedy sampling using multinomial computation and unite the codepaths. This
        has implications on the overall design of the sampler, e.g. how to record
        accurate logprobs for the user, so this improvement is deferred to later.
        """
    # NOTE: logprobs are not modified so they can be returned to the user.
    # print size of probs and sample_indices
    probs[sample_indices, :] = 0
    probs[sample_indices, greedy_samples] = 1.0
