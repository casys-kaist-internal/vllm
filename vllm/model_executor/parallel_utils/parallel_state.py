# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import torch
from typing import Optional


class ParallelState:
    """state for parallelism """

    def __init__(
        self
    ) -> None:
        # Intra-layer model parallel group that the current rank belongs to.
        self.tensor_model_parallel_group = None
        # Inter-layer model parallel group that the current rank belongs to.
        self.pipeline_model_parallel_group = None
        # Model parallel group (both intra- and pipeline) that the current rank belongs to.
        self.model_parallel_group = None
        # Embedding group.
        self.embedding_group = None
        # Position embedding group.
        self.position_embedding_group = None
        # Data parallel group that the current rank belongs to.
        self.data_parallel_group = None

        self.virtual_pipeline_model_parallel_rank = None
        self.virtual_pipeline_model_parallel_world_size = None
        self.pipeline_model_parallel_split_rank = None

        # These values enable us to change the mpu sizes on the fly.
        self.mpu_tensor_model_parallel_world_size = None
        self.mpu_pipeline_model_parallel_world_size = None
        self.mpu_tensor_model_parallel_rank = None
        self.mpu_pipeline_model_parallel_rank = None

        # A list of ranks that have a copy of the embedding.
        self.embedding_global_ranks = None

        # A list of ranks that have a copy of the position embedding.
        self.position_embedding_global_ranks = None

        # A list of global ranks for each pipeline group to ease calculation of the source
        # rank when broadcasting from the first or last pipeline stage.
        self.pipeline_global_ranks = None

        # A list of global ranks for each data parallel group to ease calculation of the source
        # rank when broadcasting weights from src to all other data parallel ranks
        self.data_parallel_global_ranks = None

    def initialize_model_parallel(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
    ) -> None:
        """
        Initialize model data parallel groups.

        Arguments:
            tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
            pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
            virtual_pipeline_model_parallel_size: number of virtual stages (interleaved
                                                pipeline).
            pipeline_model_parallel_split_rank: for models with both encoder and decoder,
                                                rank in pipeline with split point.

        Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
        use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
        the model pipeline. The present function will
        create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
        and 8 data-parallel groups as:
            8 data_parallel groups:
                [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
            8 tensor model-parallel groups:
                [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
            4 pipeline model-parallel groups:
                [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        Note that for efficiency, the caller should make sure adjacent ranks
        are on the same DGX box. For example if we are using 2 DGX-1 boxes
        with a total of 16 GPUs, rank 0 to 7 belong to the first box and
        ranks 8 to 15 belong to the second box.
        """
        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()

        if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
            )

        data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                                 pipeline_model_parallel_size)

        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        num_data_parallel_groups: int = world_size // data_parallel_size  # ???

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 2:
                raise RuntimeError("pipeline-model-parallel size should be greater than 2 with "
                                   "interleaved schedule")
            self.virtual_pipeline_model_parallel_rank = 0
            self.virtual_pipeline_model_parallel_world_size = virtual_pipeline_model_parallel_size

        if pipeline_model_parallel_split_rank is not None:
            self.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        # Build the data-parallel groups.
        assert self.data_parallel_group is None, 'data parallel group is already initialized'
        all_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank,
                              tensor_model_parallel_size)
                all_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    self.data_parallel_group = group
                    self.data_parallel_global_ranks = ranks

        # Build the model-parallel groups.
        assert self.model_parallel_group is None, 'model parallel group is already initialized'
        for i in range(data_parallel_size):
            ranks = [data_parallel_group_ranks[i]
                     for data_parallel_group_ranks in all_data_parallel_group_ranks]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                self.model_parallel_group = group

        # Build the tensor model-parallel groups.
        assert self.tensor_model_parallel_group is None, \
            'tensor model parallel group is already initialized'
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size,
                          (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                self.tensor_model_parallel_group = group

        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        assert self.pipeline_model_parallel_group is None, \
            'pipeline model parallel group is already initialized'
        assert self.embedding_group is None, 'embedding group is already initialized'
        assert self.position_embedding_group is None, \
            'position embedding group is already initialized'
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                self.pipeline_model_parallel_group = group
                self.pipeline_global_ranks = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [ranks[0],
                                           ranks[pipeline_model_parallel_split_rank],
                                           ranks[-1]]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0],
                                                    ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                self.embedding_group = group
            if rank in ranks:
                self.embedding_global_ranks = embedding_ranks

            group = torch.distributed.new_group(position_embedding_ranks)
            if rank in position_embedding_ranks:
                self.position_embedding_group = group
            if rank in ranks:
                self.position_embedding_global_ranks = position_embedding_ranks

    def model_parallel_is_initialized(self):
        """Check if model and data parallel groups are initialized."""
        if self.tensor_model_parallel_group is None or \
                self.pipeline_model_parallel_group is None or \
                self.data_parallel_group is None:
            return False
        return True

    def get_model_parallel_group(self):
        """Get the model parallel group the caller rank belongs to."""
        assert self.model_parallel_group is not None, \
            'model parallel group is not initialized'
        return self.model_parallel_group

    def get_tensor_model_parallel_group(self):
        """Get the tensor model parallel group the caller rank belongs to."""
        assert self.tensor_model_parallel_group is not None, \
            'intra_layer_model parallel group is not initialized'
        return self.tensor_model_parallel_group

    def get_pipeline_model_parallel_group(self):
        """Get the pipeline model parallel group the caller rank belongs to."""
        assert self.pipeline_model_parallel_group is not None, \
            'pipeline_model parallel group is not initialized'
        return self.pipeline_model_parallel_group

    def get_data_parallel_group(self):
        """Get the data parallel group the caller rank belongs to."""
        assert self.data_parallel_group is not None, \
            'data parallel group is not initialized'
        return self.data_parallel_group

    def get_embedding_group(self):
        """Get the embedding group the caller rank belongs to."""
        assert self.embedding_group is not None, \
            'embedding group is not initialized'
        return self.embedding_group

    def get_position_embedding_group(self):
        """Get the position embedding group the caller rank belongs to."""
        assert self.position_embedding_group is not None, \
            'position embedding group is not initialized'
        return self.position_embedding_group

    def set_tensor_model_parallel_world_size(self, world_size):
        """Set the tensor model parallel size"""
        self.mpu_tensor_model_parallel_world_size = world_size

    def set_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self.mpu_pipeline_model_parallel_world_size = world_size

    def get_tensor_model_parallel_world_size(self):
        """Return world size for the tensor model parallel group."""
        if self.mpu_tensor_model_parallel_world_size is not None:
            return self.mpu_tensor_model_parallel_world_size
        return torch.distributed.get_world_size(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_world_size(self):
        """Return world size for the pipeline model parallel group."""
        if self.mpu_pipeline_model_parallel_world_size is not None:
            return self.mpu_pipeline_model_parallel_world_size
        return torch.distributed.get_world_size(group=self.get_pipeline_model_parallel_group())

    def set_tensor_model_parallel_rank(self, rank):
        """Set tensor model parallel rank."""
        self.mpu_tensor_model_parallel_rank = rank

    def set_pipeline_model_parallel_rank(self, rank):
        """Set pipeline model parallel rank."""
        self.mpu_pipeline_model_parallel_rank = rank

    def set_pipeline_model_parallel_split_rank(self, rank):
        """Set pipeline model parallel split rank."""
        self.mpu_pipeline_model_parallel_split_rank = rank

    def get_tensor_model_parallel_rank(self):
        """Return my rank for the tensor model parallel group."""
        if self.mpu_tensor_model_parallel_rank is not None:
            return self.mpu_tensor_model_parallel_rank
        return torch.distributed.get_rank(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_rank(self):
        """Return my rank for the pipeline model parallel group."""
        if self.mpu_pipeline_model_parallel_rank is not None:
            return self.mpu_pipeline_model_parallel_rank
        return torch.distributed.get_rank(group=self.get_pipeline_model_parallel_group())

    def is_pipeline_first_stage(self, ignore_virtual=False):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            if self.get_virtual_pipeline_model_parallel_world_size() is not None and \
                    self.get_virtual_pipeline_model_parallel_rank() != 0:
                return False
        return self.get_pipeline_model_parallel_rank() == 0

    def is_pipeline_last_stage(self, ignore_virtual=False):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = \
                self.get_virtual_pipeline_model_parallel_world_size()
            if virtual_pipeline_model_parallel_world_size is not None and \
                self.get_virtual_pipeline_model_parallel_rank() != (
                    virtual_pipeline_model_parallel_world_size - 1):
                return False
        return self.get_pipeline_model_parallel_rank() == (
            self.get_pipeline_model_parallel_world_size() - 1)

    def is_rank_in_embedding_group(self, ignore_virtual=False):
        """Return true if current rank is in embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if ignore_virtual:
            return rank in self.embedding_global_ranks
        if rank in self.embedding_global_ranks:
            if rank == self.embedding_global_ranks[0]:
                return self.is_pipeline_first_stage(ignore_virtual=False)
            elif rank == self.embedding_global_ranks[-1]:
                return self.is_pipeline_last_stage(ignore_virtual=False)
            else:
                return True
        return False

    def is_rank_in_position_embedding_group(self):
        """Return true if current rank is in position embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        return rank in self.position_embedding_global_ranks

    def is_pipeline_stage_before_split(self, rank=None):
        """Return True if pipeline stage executes encoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size() == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank()
        if self.pipeline_model_parallel_split_rank is None:
            return True
        if rank < self.pipeline_model_parallel_split_rank:
            return True
        return False

    def is_pipeline_stage_after_split(self, rank=None):
        """Return True if pipeline stage executes decoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size() == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank()
        if self.pipeline_model_parallel_split_rank is None:
            return True
        if rank >= self.pipeline_model_parallel_split_rank:
            return True
        return False

    def is_pipeline_stage_at_split(self):
        """Return true if pipeline stage executes decoder block and next
        stage executes encoder block for a model with both encoder and
        decoder."""
        rank = self.get_pipeline_model_parallel_rank()
        return self.is_pipeline_stage_before_split(rank) and \
            self.is_pipeline_stage_after_split(rank+1)

    def get_virtual_pipeline_model_parallel_rank(self):
        """Return the virtual pipeline-parallel rank."""
        return self.virtual_pipeline_model_parallel_rank

    def set_virtual_pipeline_model_parallel_rank(self, rank):
        """Set the virtual pipeline-parallel rank."""
        self.virtual_pipeline_model_parallel_rank = rank

    def get_virtual_pipeline_model_parallel_world_size(self):
        """Return the virtual pipeline-parallel world size."""
        return self.virtual_pipeline_model_parallel_world_size

    def get_tensor_model_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the tensor model parallel group."""
        global_rank = torch.distributed.get_rank()
        local_world_size = self.get_tensor_model_parallel_world_size()
        return (global_rank // local_world_size) * local_world_size

    def get_data_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the data parallel group."""
        assert self.data_parallel_global_ranks is not None, \
            "Data parallel group is not initialized"
        return self.data_parallel_global_ranks[0]

    def get_pipeline_model_parallel_first_rank(self):
        """Return the global rank of the first process in the pipeline for the
        current tensor parallel group"""
        assert self.pipeline_global_ranks is not None, \
            "Pipeline parallel group is not initialized"
        return self.pipeline_global_ranks[0]

    def get_pipeline_model_parallel_last_rank(self):
        """Return the global rank of the last process in the pipeline for the
        current tensor parallel group"""
        assert self.pipeline_global_ranks is not None, \
            "Pipeline parallel group is not initialized"
        last_rank_local = self.get_pipeline_model_parallel_world_size() - 1
        return self.pipeline_global_ranks[last_rank_local]

    def get_pipeline_model_parallel_next_rank(self):
        """Return the global rank that follows the caller in the pipeline"""
        assert self.pipeline_global_ranks is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self.pipeline_global_ranks[(rank_in_pipeline + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self):
        """Return the global rank that preceeds the caller in the pipeline"""
        assert self.pipeline_global_ranks is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self.pipeline_global_ranks[(rank_in_pipeline - 1) % world_size]

    def get_data_parallel_world_size(self):
        """Return world size for the data parallel group."""
        return torch.distributed.get_world_size(group=self.get_data_parallel_group())

    def get_data_parallel_rank(self):
        """Return my rank for the data parallel group."""
        return torch.distributed.get_rank(group=self.get_data_parallel_group())

    def destroy_model_parallel(self):
        """Set the groups to none."""
        self.model_parallel_group = None
        self.tensor_model_parallel_group = None
        self.pipeline_model_parallel_group = None
        self.data_parallel_group = None
        self.embedding_group = None
        self.position_embedding_group = None
        self.virtual_pipeline_model_parallel = None
        self.virtual_pipeline_model_parallel_world_size = None
        self.mpu_tensor_model_parallel_world_size = None
        self.mpu_pipeline_model_parallel_world_size = None
        self.tensor_model_parallel_rank = None
        self.mpu_pipeline_model_parallel_rank = None
