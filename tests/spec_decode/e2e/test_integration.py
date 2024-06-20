"""Tests which cover integration of the speculative decoding framework with
other features, e.g. cuda graphs.
"""

import pytest

from .conftest import run_greedy_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Verify equality when cuda graphs allowed.
        "enforce_eager": False,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "baseline_llm_kwargs",
    [{
        "model": "facebook/opt-13b",
    }])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "target_model": "facebook/opt-13b",
                             "draft_model": "facebook/opt-125m",
                             "draft_size": 5,
                             "disable_bonus_token": False,
                             "target_attention": True,
                         }])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("output_len", [128])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_cuda_graph(baseline_llm_generator, test_llm_generator,
                                batch_size, output_len):
    """Verify spec decode equality when cuda graphs are enabled.
    """
    run_greedy_equality_correctness_test(
        baseline_llm_generator,
        test_llm_generator,
        batch_size,
        max_output_len=output_len,
        force_output_len=True,
    )
