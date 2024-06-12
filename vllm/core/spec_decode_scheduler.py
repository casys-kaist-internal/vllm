import enum
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union, OrderedDict, Set
from dataclasses import dataclass, field

from vllm.config import CacheConfig, SchedulerConfig, SpecDecodeConfig
from vllm.core.spec_decode_block_manager import SpecDecodeAllocStatus, SpecDecodeBlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus, SpecDecodeStage)
from vllm.utils import nvtx_range

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int
    do_sample: bool = True


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _requeset_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _requeset_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            return

        self._requeset_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._requeset_ids_num_batched_tokens:
            self._requeset_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            return

        self._requeset_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._requeset_ids_num_curr_seqs:
            self._requeset_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


class SpecDecodeSchedulerOutputs:

    def __init__(
        self,
        prefill_scheduled_seq_groups: List[SequenceGroup],
        target_decode_scheduled_seq_groups: List[SequenceGroup],
        draft_decode_scheduled_seq_groups: List[SequenceGroup],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.prefill_scheduled_seq_groups = prefill_scheduled_seq_groups
        self.target_decode_scheduled_seq_groups = target_decode_scheduled_seq_groups
        self.draft_decode_scheduled_seq_groups = draft_decode_scheduled_seq_groups
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

        if self.prefill_scheduled_seq_groups or self.target_decode_scheduled_seq_groups:
            self.is_target = True
        else:
            self.is_target = False

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.prefill_scheduled_seq_groups and not self.target_decode_scheduled_seq_groups
                and not self.draft_decode_scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class SpecDecodeScheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        spec_decode_config: SpecDecodeConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.spec_decode_config = spec_decode_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = SpecDecodeBlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []

        # NOTE(sangjin): We dont consider self.running queue. We split the running queue into draft and target
        self.need_to_run_draft: List[SequenceGroup] = []
        self.need_to_run_target: List[SequenceGroup] = []

        self.balancing_queue: List[SequenceGroup] = []

        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.need_to_run_draft, self.need_to_run_target, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.need_to_run_draft or self.need_to_run_target or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.need_to_run_draft) + len(self.need_to_run_target) + len(self.swapped)

    def get_num_running_seq_groups(self) -> int:
        return len(self.need_to_run_draft) + len(self.need_to_run_target) + len(self.balancing_queue)

    def swap_target_draft_queues(self, scheduler_outputs) -> None:
        scheduled_seq_groups = scheduler_outputs.prefill_scheduled_seq_groups + \
            scheduler_outputs.target_decode_scheduled_seq_groups + \
            scheduler_outputs.draft_decode_scheduled_seq_groups

        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            if seq_group.spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
                self.need_to_run_draft.remove(seq_group)
                self.need_to_run_target.append(seq_group)
            else:
                self.need_to_run_target.remove(seq_group)
                self.need_to_run_draft.append(seq_group)

    def _balance_target_draft_queues(self) -> None:
        # Need to balance the draft and target queues
        # the balancing queue is used to balance the draft and target queues
        # Steps:
        # 1) If balancing queue is not empty, move sequence groups from balancing queue to draft or target queue
        # 2) Access current load
        # 3) Determine imbalance
        # 4) Rebalance queues by moving sequence groups to balancing queue

        # Move sequence groups from balancing queue back to appropriate queues after one step
        i = 0
        while i < len(self.balancing_queue):
            seq_group = self.balancing_queue[i]

            if seq_group.spec_decode_stage == SpecDecodeStage.DRAFT_DECODE:
                self.need_to_run_target.append(seq_group)
            else:
                self.need_to_run_draft.append(seq_group)

            # Remove the processed sequence group from the balancing queue
            del self.balancing_queue[i]

        # Assess current load
        draft_load = len(self.need_to_run_draft)
        target_load = len(self.need_to_run_target)
        total_load = draft_load + target_load
        balance_point = total_load // 2

        if total_load > 0:
            imbalance_percentage = abs(
                draft_load - target_load) / total_load * 100
        else:
            imbalance_percentage = 0

        # Determine imbalance
        if total_load < 32 or imbalance_percentage <= self.scheduler_config.balance_threshold:
            return  # Queues are balanced within the acceptable range

        # Rebalance queues
        if draft_load > target_load:
            # Move sequence groups from draft to balancing queue
            while len(self.need_to_run_draft) > balance_point:
                seq_group = self.need_to_run_draft.pop()
                self.balancing_queue.append(seq_group)
        else:
            # Move sequence groups from target to balancing queue
            while len(self.need_to_run_target) > balance_point:
                seq_group = self.need_to_run_target.pop()
                self.balancing_queue.append(seq_group)

        logger.info(
            f"Rebalanced queues: {len(self.need_to_run_draft)} draft, {len(self.need_to_run_target)} target, {len(self.balancing_queue)} in balancing queue")

    def _schedule_prefill(self,
                          scheduled_seq_groups: List[ScheduledSequenceGroup],
                          ignored_seq_groups: List[SequenceGroup],
                          budget: SchedulingBudget):
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                            for seq_group in self.need_to_run_draft)
        num_curr_seqs += sum(seq_group.get_max_num_running_seqs()
                             for seq_group in self.need_to_run_target)

        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq_group = self.waiting[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            seq = waiting_seqs[0]
            num_new_tokens = seq.get_num_uncomputed_target_tokens()

            do_sample = True
            if self.scheduler_config.chunk_prefill_enabled:
                if num_new_tokens > budget.remaining_token_budget():
                    num_new_tokens = budget.remaining_token_budget()
                    do_sample = False

            if num_new_tokens > self.prompt_limit:
                logger.warning(
                    f"Input prompt ({num_new_tokens} tokens) is too long"
                    f" and exceeds limit of {self.prompt_limit}")
                seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == SpecDecodeAllocStatus.LATER:
                break
            elif can_allocate == SpecDecodeAllocStatus.NEVER:
                logger.warning(
                    f"Input prompt ({num_new_tokens} tokens) is too long"
                    f" and exceeds the capacity of block_manager")
                seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                self.waiting.pop(0)
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                           num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request
            seq_group = self.waiting.pop(0)
            self._allocate(seq_group)
            num_curr_seqs += num_new_seqs
            self.need_to_run_target.append(seq_group)
            seq_group.spec_decode_stage = SpecDecodeStage.PREFILL
            budget.add_num_batched_tokens(
                seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
            scheduled_seq_groups.append(
                ScheduledSequenceGroup(seq_group, num_new_tokens, do_sample))

    def _schedule_draft_decode(self,
                               scheduled_seq_groups: List[ScheduledSequenceGroup],
                               blocks_to_swap_in: Dict[int, int],
                               blocks_to_swap_out: Dict[int, int],
                               blocks_to_copy: Dict[int, int]):
        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        now = time.time()
        self.need_to_run_draft = self.policy.sort_by_priority(
            now, self.need_to_run_draft)

        # Reserve new token slots for the running sequence groups.
        preempted: List[SequenceGroup] = []
        need_to_run_draft: List[SequenceGroup] = []
        while self.need_to_run_draft:
            seq_group = self.need_to_run_draft.pop(0)
            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            # We reserve new token slots also considering the target decode stage.
            while not self.block_manager.can_append_slots(seq_group, seq.draft_size + 1):
                if self.need_to_run_draft:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.need_to_run_draft.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Can schedule this request
                self._append_slots(
                    seq_group, seq.draft_size + 1, blocks_to_copy)
                need_to_run_draft.append(seq_group)
                seq_group.spec_decode_stage = SpecDecodeStage.DRAFT_DECODE
                scheduled_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, seq.get_num_uncomputed_draft_tokens()))

        self.need_to_run_draft = need_to_run_draft

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.need_to_run_draft)
            num_curr_seqs += sum(seq_group.get_max_num_running_seqs()
                                 for seq_group in self.need_to_run_target)

            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
                self._append_slots(
                    seq_group, seq.draft_size + 1, blocks_to_copy)
                num_curr_seqs += num_new_seqs

                self.need_to_run_draft.append(seq_group)
                seq_group.spec_decode_stage = SpecDecodeStage.DRAFT_DECODE
                scheduled_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, seq.get_num_uncomputed_draft_tokens()))

    def _schedule_target_decode(self,
                                scheduled_seq_groups: List[ScheduledSequenceGroup],
                                budget: SchedulingBudget) -> int:
        # Sort the sequence groups by priority.
        now = time.time()
        self.need_to_run_target = self.policy.sort_by_priority(
            now, self.need_to_run_target)

        # We don't consider preemption for target decode because we already
        # considered appending slots including the target token (also the bonus token)
        # in the draft decoding stage.
        for seq_group in self.need_to_run_target:
            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            assert seq.draft_size == seq.get_draft_len()
            seq_group.spec_decode_stage = SpecDecodeStage.TARGET_DECODE
            budget.add_num_batched_tokens(
                seq_group.request_id, (seq.get_draft_len() + 1))
            scheduled_seq_groups.append(
                ScheduledSequenceGroup(seq_group, seq.get_num_uncomputed_target_tokens()))

    def _schedule(self) -> SpecDecodeSchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        prefill_scheduled_seq_groups: List[ScheduledSequenceGroup] = []
        target_decode_scheduled_seq_groups: List[ScheduledSequenceGroup] = []
        draft_decode_scheduled_seq_groups: List[ScheduledSequenceGroup] = []

        # Ignore the sequence groups that are too long.
        ignored_seq_groups: List[SequenceGroup] = []

        # Inlcude running requests to the budget.
        target_budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        for seq_group in self.need_to_run_target:
            target_budget.add_num_seqs(seq_group.request_id,
                                       seq_group.get_max_num_running_seqs())

        if not self.scheduler_config.chunk_prefill_enabled:
            # Join waiting sequences if possible.
            if not self.swapped:
                self._schedule_prefill(
                    prefill_scheduled_seq_groups, ignored_seq_groups, target_budget)

            if target_budget.num_batched_tokens == 0:
                if self.need_to_run_draft:
                    self._schedule_draft_decode(draft_decode_scheduled_seq_groups,
                                                blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

                else:
                    self._schedule_target_decode(
                        target_decode_scheduled_seq_groups, target_budget)

        else:
            if self.need_to_run_draft:
                self._schedule_draft_decode(draft_decode_scheduled_seq_groups,
                                            blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

            else:
                self._schedule_target_decode(
                    target_decode_scheduled_seq_groups, target_budget)
                self._schedule_prefill(
                    prefill_scheduled_seq_groups, ignored_seq_groups, target_budget)

        scheduler_outputs = SpecDecodeSchedulerOutputs(
            prefill_scheduled_seq_groups=prefill_scheduled_seq_groups,
            target_decode_scheduled_seq_groups=target_decode_scheduled_seq_groups,
            draft_decode_scheduled_seq_groups=draft_decode_scheduled_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
        )
        return scheduler_outputs

    def _collocate_schedule(self) -> Tuple[SpecDecodeSchedulerOutputs, SpecDecodeSchedulerOutputs]:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        prefill_scheduled_seq_groups: List[ScheduledSequenceGroup] = []
        target_decode_scheduled_seq_groups: List[ScheduledSequenceGroup] = []
        draft_decode_scheduled_seq_groups: List[ScheduledSequenceGroup] = []

        # Ignore the sequence groups that are too long.
        ignored_seq_groups: List[SequenceGroup] = []

        # Inlcude running requests to the budget.
        target_budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        for seq_group in self.need_to_run_target:
            target_budget.add_num_seqs(seq_group.request_id,
                                       seq_group.get_max_num_running_seqs())

        # Target prefill + decoding scheduling
        if not self.scheduler_config.chunk_prefill_enabled:
            # Schedule target prefill or target decoding
            # Join waiting sequences if possible.
            if not self.swapped:
                self._schedule_prefill(
                    prefill_scheduled_seq_groups, ignored_seq_groups, target_budget)

            if target_budget.num_batched_tokens == 0:
                self._schedule_target_decode(
                    target_decode_scheduled_seq_groups, target_budget)

        else:
            # Schedule target prefill and decoding together
            self._schedule_target_decode(
                target_decode_scheduled_seq_groups, target_budget)
            self._schedule_prefill(prefill_scheduled_seq_groups, ignored_seq_groups,
                                   target_budget)

        # Draft decode scheduling
        self._schedule_draft_decode(draft_decode_scheduled_seq_groups,
                                    blocks_to_swap_in,
                                    blocks_to_swap_out,
                                    blocks_to_copy)

        target_scheduler_outputs = SpecDecodeSchedulerOutputs(
            prefill_scheduled_seq_groups=prefill_scheduled_seq_groups,
            target_decode_scheduled_seq_groups=target_decode_scheduled_seq_groups,
            draft_decode_scheduled_seq_groups=[],
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            ignored_seq_groups=ignored_seq_groups,
        )
        draft_scheduler_outputs = SpecDecodeSchedulerOutputs(
            prefill_scheduled_seq_groups=[],
            target_decode_scheduled_seq_groups=[],
            draft_decode_scheduled_seq_groups=draft_decode_scheduled_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return target_scheduler_outputs, draft_scheduler_outputs

    @ nvtx_range("SpecDecodeScheduler::schedule")
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], List[SequenceGroupMetadata],
                                List[SequenceGroupMetadata], SpecDecodeSchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.need_to_run_target, self.need_to_run_draft, self.swapped, and self.waiting.

        scheduler_outputs = self._schedule()

        prefill_seq_group_metadata_list = self._create_seq_group_metadata_list(
            scheduler_outputs.prefill_scheduled_seq_groups)
        target_decode_seq_group_metadata_list = self._create_seq_group_metadata_list(
            scheduler_outputs.target_decode_scheduled_seq_groups)
        draft_decode_seq_group_metadata_list = self._create_seq_group_metadata_list(
            scheduler_outputs.draft_decode_scheduled_seq_groups)

        return (prefill_seq_group_metadata_list, target_decode_seq_group_metadata_list,
                draft_decode_seq_group_metadata_list, scheduler_outputs)

    @ nvtx_range("SpecDecodeScheduler::collocate_schedule")
    def collocate_schedule(self) -> Tuple[List[SequenceGroupMetadata], List[SequenceGroupMetadata],
                                          List[SequenceGroupMetadata], SpecDecodeSchedulerOutputs, SpecDecodeSchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.need_to_run_target, self.need_to_run_draft, self.swapped, and self.waiting.
        # Balance the draft and target queues
        self._balance_target_draft_queues()

        target_scheduler_outputs, draft_scheduler_outputs = self._collocate_schedule()

        prefill_seq_group_metadata_list = self._create_seq_group_metadata_list(
            target_scheduler_outputs.prefill_scheduled_seq_groups)
        target_decode_seq_group_metadata_list = self._create_seq_group_metadata_list(
            target_scheduler_outputs.target_decode_scheduled_seq_groups)
        draft_decode_seq_group_metadata_list = self._create_seq_group_metadata_list(
            draft_scheduler_outputs.draft_decode_scheduled_seq_groups)

        return (prefill_seq_group_metadata_list, target_decode_seq_group_metadata_list,
                draft_decode_seq_group_metadata_list, target_scheduler_outputs, draft_scheduler_outputs)

    def _create_seq_group_metadata_list(
        self,
        scheduled_seq_groups: List[ScheduledSequenceGroup]
    ) -> List[SequenceGroupMetadata]:
        seq_group_metadata_list: List[SequenceGroupMetadata] = []

        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size

            spec_decode_stage = seq_group.spec_decode_stage
            seq_data: OrderedDict[int, SequenceData] = OrderedDict()
            block_tables: Dict[int, List[int]] = {}

            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            seq_id = seq.seq_id
            seq_data[seq_id] = seq.data
            block_tables[seq_id] = self.block_manager.get_block_table(seq)
            is_prompt = True if spec_decode_stage == SpecDecodeStage.PREFILL else False

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                token_chunk_size=token_chunk_size,
                spec_decode_stage=spec_decode_stage,
                draft_size=seq.draft_size
            )
            seq_group_metadata_list.append(seq_group_metadata)

        return seq_group_metadata_list

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.need_to_run_draft = [
            seq_group for seq_group in self.need_to_run_draft
            if not seq_group.is_finished()
        ]

        self.need_to_run_target = [
            seq_group for seq_group in self.need_to_run_target
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        size: int,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slots(seq, size)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
            seq.reset_state_for_recompute()

        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
