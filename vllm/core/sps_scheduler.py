import enum
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, SchedulerConfig, SpSConfig
from vllm.core.sps_block_manager import SpSAllocStatus, SpSBlockSpaceManager
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.core.sps_util import find_optimal_draft_size
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus, SpSStage)

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


class SpSSchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        sps_stage: SpSStage,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.sps_stage = sps_stage
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class SpSScheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        sps_config: SpSConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.sps_config = sps_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = SpSBlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

        # NOTE(sangjin): We dont consider self.running queue. We split the running queue into draft and target
        self.need_to_run_draft: List[SequenceGroup] = []
        self.need_to_run_target: List[SequenceGroup] = []


    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        def clear_state_queue(
            state_queue: List[SequenceGroup], request_ids: Set[str]
        ) -> List[SequenceGroup]:
            if len(request_ids) == 0:
                return state_queue
            to_add = []
            for seq_group in state_queue:
                if seq_group.request_id in request_ids:
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return to_add
                else:
                    to_add.append(seq_group)
            return to_add

        request_ids = {request_id} if isinstance(request_id, str) else set(request_id)
        self.waiting = clear_state_queue(self.waiting, request_ids)
        self.running = clear_state_queue(self.running, request_ids)
        self.swapped = clear_state_queue(self.swapped, request_ids)

    def abort_all_seq_groups(self) -> None:
        self.abort_seq_group(
            [
                seq_group.request_id
                for seq_group in self.waiting + self.running + self.swapped
            ]
        )

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.draft_exit

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.draft_exit)

    def _multi_step_schedule(self) -> SpSSchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        # now = time.monotonic()

        # PROMPT PHASE START 
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds the capacity of block_manager")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                
                # Instead of appending to running queue, we divide the running queue into draft and target
                # self.running.append(seq_group)
                self.need_to_run_draft(seq_group)
                
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SpSSchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    sps_stage=SpSStage.PROMPT,
                    num_batched_tokens=len(seq_lens) *
                    max(seq_lens) if seq_lens else 0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs
        # PROMPT PHASE END

        # Decision logic for executing the target or draft model:
        if self.need_to_run_draft:
            # DRAFT_DECODING PHASE START
            sps_stage = SpSStage.DRAFT_DECODE
            
            if self.sps_config.use_dynamic_draft_size:
                find_optimal_draft_size(self.need_to_run_draft, self.sps_config)                 

            # NOTE(woosuk): Preemption happens only when there is no available slot
            # to keep all the sequence groups in the RUNNING state.
            # In this case, the policy is responsible for deciding which sequence
            # groups to preempt.
            # self.running = self.policy.sort_by_priority(now, self.running)

            # FIXME(sangjin): How to handle case of draft preemption? Should we not
            # allow draft preemption at all?

            # Reserve new token slots for the running sequence groups.
            need_to_run_draft: List[SequenceGroup] = []
            while self.need_to_run_draft:
                seq_group = self.need_to_run_draft.pop(0)
                seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

                # Simplify preemption logic
                if not self.block_manager.can_append_slots(seq_group, seq.draft_size):
                    raise AssertionError("Draft preemption is not supported.")
                else:
                    # Append new slots to the sequence group.
                    self._append_slots(seq_group, seq.draft_size, blocks_to_copy)
                    need_to_run_draft.append(seq_group)
            self.need_to_run_draft = need_to_run_draft

            if self.need_to_run_draft:
                num_batched_tokens = 0
                for seq_group in self.need_to_run_draft:
                    seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
                    num_batched_tokens += seq.data.get_uncached_draft_len()

                scheduler_outputs = SpSSchedulerOutputs(
                    scheduled_seq_groups=self.need_to_run_draft,
                    sps_stage=sps_stage,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=[],
                )
                return scheduler_outputs
            # DRAFT_DECODING PHASE END
        else: 
            # TARGET_DECODING PHASE START
            sps_stage = SpSStage.TARGET_DECODE

            # NOTE(woosuk): Preemption happens only when there is no available slot
            # to keep all the sequence groups in the RUNNING state.
            # In this case, the policy is responsible for deciding which sequence
            # groups to preempt.
            # self.running = self.policy.sort_by_priority(now, self.running)

            # FIXME(sangjin): How to handle case of target preemption?
            # Simplify preemption logic
            need_to_run_target: List[SequenceGroup] = []
            while self.need_to_run_target:
                seq_group = self.need_to_run_target.pop(0)
                seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

                # Simplify preemption logic
                if not self.block_manager.can_append_slots(seq_group, 1):
                    raise AssertionError("Target preemption is not supported.")
                else:
                    # Append new slots to the sequence group.
                    self._append_slots(seq_group, 1, blocks_to_copy)
                    need_to_run_target.append(seq_group)
            self.need_to_run_target = need_to_run_target

            num_batched_tokens = 0
            for seq_group in self.need_to_run_target:
                seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                assert len(seqs) == 1, "SpS does not support beam search"
                # target model should run the last non-draft token and all the draft tokens
                num_batched_tokens += (seqs[0].get_draft_len() + 1)

            # print("num_batched_tokens: ", num_batched_tokens)
            if self.sps_config.get_tile_size_constraint(len(self.running)) < num_batched_tokens:
                raise AssertionError("Tile size constraint is violated.")

            scheduler_outputs = SpSSchedulerOutputs(
                scheduled_seq_groups=self.need_to_run_target,
                sps_stage=sps_stage,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=[],
            )

            return scheduler_outputs

    def _schedule(self) -> SpSSchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.monotonic()

        # Target Execution Point Identifier (TEPI)
        # Count the number of tokens that should be fed to the target model.
        num_tokens_to_target = 0
        for seq_group in self.running:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            # target model should run the last non-draft token and all the draft tokens
            num_tokens_to_target += (seqs[0].get_draft_len() + 1)

        num_draft_exit_tokens_to_target = 0
        for seq_group in self.draft_exit:
            seqs = seq_group.get_seqs(status=SequenceStatus.DRAFT_EXIT)
            # target model should run the last non-draft token and all the draft tokens
            num_draft_exit_tokens_to_target += (seqs[0].get_draft_len() + 1)

        num_tokens_to_target += num_draft_exit_tokens_to_target
        # print("num_tokens_to_target_draft_exit: ",
        #       num_tokens_to_target_draft_exit)
        # print("num_tokens_to_target: ", num_tokens_to_target)

        # Calculate the number of tokens that can be additionally fed to the target model.
        remaining_tokens = self.sps_config.get_tile_size_constraint(len(self.running)) - num_tokens_to_target

        # PROMPT PHASE START
        # Join waiting sequences if possible.
        if not self.swapped and remaining_tokens > 0:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting and remaining_tokens > 0:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds the capacity of block_manager")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)
                remaining_tokens -= 1

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SpSSchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    sps_stage=SpSStage.PROMPT,
                    num_batched_tokens=len(seq_lens) *
                    max(seq_lens) if seq_lens else 0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs
        # PROMPT PHASE END

        # Decision logic for executing the target or draft model:
        # With respect to the threshold, if more tokens can be fed to the target model,
        # continue running the draft model. Otherwise, switch to the target model.
        run_target_model = (remaining_tokens <= 0) or (len(self.running) == 0)

        if not run_target_model:
            sps_stage = SpSStage.DRAFT_DECODE

            # NOTE(woosuk): Preemption happens only when there is no available slot
            # to keep all the sequence groups in the RUNNING state.
            # In this case, the policy is responsible for deciding which sequence
            # groups to preempt.
            self.running = self.policy.sort_by_priority(now, self.running)

            # FIXME(sangjin): How to handle case of draft preemption? Should we not
            # allow draft preemption at all?

            # Reserve new token slots for the running sequence groups.
            running: List[SequenceGroup] = []
            while self.running:
                seq_group = self.running.pop(0)
                seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

                if seq.get_draft_len() >= self.sps_config.draft_size:
                    self.draft_exit.append(seq_group)
                    seq.status = SequenceStatus.DRAFT_EXIT
                elif seq.get_len() + seq.get_draft_len() >= self.scheduler_config.max_model_len:
                    self.draft_exit.append(seq_group)
                    seq.status = SequenceStatus.DRAFT_EXIT
                elif remaining_tokens > 0:
                    while not self.block_manager.can_append_slot(seq_group):
                        if self.running:
                            # Preempt the lowest-priority sequence groups.
                            victim_seq_group = self.running.pop(-1)
                            self._preempt(victim_seq_group, blocks_to_swap_out)
                        else:
                            # No other sequence groups can be preempted.
                            # Preempt the current sequence group.
                            self._preempt(seq_group, blocks_to_swap_out)
                            break
                    else:
                        # Append new slots to the sequence group.
                        self._append_slot(seq_group, blocks_to_copy)
                        running.append(seq_group)
                        remaining_tokens -= 1
                else:
                    self.draft_exit.append(seq_group)
                    seq.status = SequenceStatus.DRAFT_EXIT
            self.running = running

            if self.running:
                # Each sequence in the generation phase only takes one token slot.
                # Therefore, the number of batched tokens is equal to the number of
                # sequences in the RUNNING state.
                num_batched_tokens = 0
                for seq_group in self.running:
                    seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
                    num_batched_tokens += seq.data.get_uncached_draft_len()

                scheduler_outputs = SpSSchedulerOutputs(
                    scheduled_seq_groups=self.running,
                    sps_stage=sps_stage,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=[],
                )
                return scheduler_outputs
            else:
                run_target_model = True

        if run_target_model:
            sps_stage = SpSStage.TARGET_DECODE

            # if self.sps_config.get_num_tokens_to_target_threshold(len(self.running)) < num_tokens_to_target:
            #     overflow = num_tokens_to_target - \
            #         self.sps_config.get_num_tokens_to_target_threshold(
            #             len(self.running))
            #     print("!!!OVERFLOW ", overflow)

            # Change the status of the sequence groups in the DRAFT_EXIT state to the RUNNING state.
            while self.draft_exit:
                seq_group = self.draft_exit.pop(0)
                seqs = seq_group.get_seqs(status=SequenceStatus.DRAFT_EXIT)
                seqs[0].status = SequenceStatus.RUNNING
                self.running.append(seq_group)

            # NOTE(woosuk): Preemption happens only when there is no available slot
            # to keep all the sequence groups in the RUNNING state.
            # In this case, the policy is responsible for deciding which sequence
            # groups to preempt.
            self.running = self.policy.sort_by_priority(now, self.running)

            # FIXME(sangjin): How to handle case of target preemption?

            # Reserve new token slots for the running sequence groups.
            running: List[SequenceGroup] = []
            while self.running:
                seq_group = self.running.pop(0)
                while not self.block_manager.can_append_slot(seq_group):
                    if self.running:
                        # Preempt the lowest-priority sequence groups.
                        victim_seq_group = self.running.pop(-1)
                        self._preempt(victim_seq_group, blocks_to_swap_out)
                    else:
                        # No other sequence groups can be preempted.
                        # Preempt the current sequence group.
                        self._preempt(seq_group, blocks_to_swap_out)
                        break
                else:
                    # Append new slots to the sequence group.
                    self._append_slot(seq_group, blocks_to_copy)
                    running.append(seq_group)

            self.running = running

            num_batched_tokens = 0
            for seq_group in self.running:
                seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
                assert len(seqs) == 1, "SpS does not support beam search"
                # target model should run the last non-draft token and all the draft tokens
                num_batched_tokens += (seqs[0].get_draft_len() + 1)

            # print("num_batched_tokens: ", num_batched_tokens)

            scheduler_outputs = SpSSchedulerOutputs(
                scheduled_seq_groups=self.running,
                sps_stage=sps_stage,
                num_batched_tokens=num_batched_tokens,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                ignored_seq_groups=[],
            )

            return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SpSSchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._multi_step_schedule()
        # scheduler_outputs = self._schedule()

        # print("Number of scheduled seq groups: ", len(
        #     scheduler_outputs.scheduled_seq_groups), scheduler_outputs.sps_stage)

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            draft_size = 0
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                draft_size = seq.draft_size

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=True if scheduler_outputs.sps_stage == SpSStage.PROMPT else False,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                sps_stage=scheduler_outputs.sps_stage,
                draft_size=draft_size,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    # def _allocate(self, seq_group: SequenceGroup, size: int) -> None:
    #     self.block_manager.allocate(seq_group, size)
    #     for seq in seq_group.get_seqs():
    #         seq.status = SequenceStatus.RUNNING

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

        # Currently we don't support preemption raise assertion
        raise AssertionError("Invalid preemption mode.")

        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        # print("preemption_mode: ", preemption_mode)
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
            # Discard all draft tokens
            seq.accept_draft_tokens(0)
            self.block_manager.free(seq)

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
        seq_group: SequenceStatus,
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
