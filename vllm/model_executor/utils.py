"""Utils for model executor."""
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from vllm.model_executor.parallel_utils.parallel_state import ParallelState
from vllm.model_executor.parallel_utils.tensor_parallel import model_parallel_cuda_manual_seed ## yh : parallel code 


def set_random_seed(seed: int, parallel_state: ParallelState) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if parallel_state.model_parallel_is_initialized():
        model_parallel_cuda_manual_seed(seed, parallel_state)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)
