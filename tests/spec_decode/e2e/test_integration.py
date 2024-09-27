"""Tests which cover integration of the speculative decoding framework with
other features, e.g., CUDA graphs.
"""

import pytest
from .conftest import run_output_distribution_similarity_test

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# Temperatures to test
temperatures = [0, 0.5, 1.0]

# Batch sizes per temperature
# When temperature is not 0, batch_size should be large enough to ensure that
# the output distribution similarity test is meaningful.
batch_sizes_per_temperature = {
    0: [1, 8],
    0.5: [128],
    1.0: [128],
}

# Gamma mapping attention options
gamma_mapping_attentions = [False, True]

# Colocate options
colocates = [False, True]

# Draft sizes
draft_sizes = [7]

# Prefill modes
prefill_modes = ["full_prefill", "chunked_prefill"]

# Output length
output_len = 256

# ---------------------------------------------------------------------------
# Generate test functions
# ---------------------------------------------------------------------------

# Helper function to generate test IDs for readability
def generate_test_id(draft_size, batch_size, prefill_mode, gamma_mapping_attention, colocate, temperature):
    return (
        f"draft{draft_size}_batch{batch_size}_"
        f"{prefill_mode}_gamma{'True' if gamma_mapping_attention else 'False'}_"
        f"colocate{'True' if colocate else 'False'}_temp{temperature}"
    )

# Iterate over all combinations and create test functions
for temperature in temperatures:
    batch_sizes = batch_sizes_per_temperature[temperature]
    for batch_size in batch_sizes:
        for draft_size in draft_sizes:
            for prefill_mode in prefill_modes:
                for gamma_mapping_attention in gamma_mapping_attentions:
                    for colocate in colocates:
                        # Capture the current values of variables in the loop
                        def make_test_function(
                            temperature=temperature,
                            batch_size=batch_size,
                            draft_size=draft_size,
                            prefill_mode=prefill_mode,
                            gamma_mapping_attention=gamma_mapping_attention,
                            colocate=colocate,
                        ):
                            # Generate a unique test function name
                            test_name = (
                                f"test_draft{draft_size}_batch{batch_size}_{prefill_mode}_"
                                f"gamma{'True' if gamma_mapping_attention else 'False'}_"
                                f"colocate{'True' if colocate else 'False'}_temp{temperature}"
                            )

                            @pytest.mark.parametrize("common_llm_kwargs", [{"enforce_eager": False}])
                            @pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
                            @pytest.mark.parametrize("baseline_llm_kwargs", [{
                                "model": "facebook/opt-6.7b",
                                "max_num_batched_tokens": 2048,
                                "max_num_seqs": 128,
                            }])
                            @pytest.mark.parametrize("test_llm_kwargs", [{
                                "target_model": "facebook/opt-6.7b",
                                "draft_model": "facebook/opt-125m",
                                "draft_size": draft_size,
                                "gamma_mapping_attention": gamma_mapping_attention,
                                "max_num_batched_tokens": 2048 if prefill_mode == "full_prefill" else 10,
                                "max_num_seqs": 10 if prefill_mode == "chunked_prefill" else 128,
                                "colocate": colocate,
                                "prefill_schedule_mode": prefill_mode
                            }])
                            @pytest.mark.parametrize("seed", [1])
                            def test_function(
                                baseline_llm_generator,
                                test_llm_generator,
                                batch_size=batch_size,
                                output_len=output_len,
                            ):
                                """Generated test function."""
                                plot_filename = generate_test_id(
                                    draft_size, batch_size, prefill_mode, gamma_mapping_attention, colocate, temperature
                                ) + ".png"
                                run_output_distribution_similarity_test(
                                    baseline_llm_generator,
                                    test_llm_generator,
                                    temperature,
                                    batch_size,
                                    max_output_len=output_len,
                                    force_output_len=True,
                                    plot_filename=plot_filename
                                )

                            # Assign the test function to the globals with the generated name
                            globals()[test_name] = test_function

                        # Call the function to create and register the test function
                        make_test_function()