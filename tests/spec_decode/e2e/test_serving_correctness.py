import pytest

from vllm import SpecDecodeLLM

from .conftest import run_greedy_serving_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "baseline_llm_kwargs",
    [
        {
            "model": "facebook/opt-125m",
        },
        {
            "model": "facebook/opt-350m",
        },
    ],
)
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "target_model": "facebook/opt-125m",
            "draft_model": "facebook/opt-125m",
            "draft_size": 5,
        },
        {
            "target_model": "facebook/opt-350m",
            "draft_model": "facebook/opt-125m",
            "draft_size": 5,
        },
    ])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use long output len for the small model test.
        1536,
    ],
)
@pytest.mark.parametrize("num_requests", [512])
@pytest.mark.parametrize("request_rate", [32, 128, 256])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_bs1(
        baseline_llm_generator, test_llm_generator, num_requests: int,
        request_rate: float, output_len: int):
    """Verify greedy equality on a tiny model with batch size of one.

    Since this test is cheaper than other e2e correctness tests, we generate
    with a higher output_len.
    """
    run_greedy_serving_correctness_test(baseline_llm_generator,
                                        test_llm_generator,
                                        num_requests=num_requests,
                                        max_output_len=output_len,
                                        request_rate=request_rate,
                                        force_output_len=True,
                                        print_tokens=False)