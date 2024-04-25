from typing import List
from vllm.config import SpSConfig
from vllm.sequence import (SequenceGroup, SequenceStatus)

def compute_value(beta, k):
    """ Compute the additional value added by `k` instances of an item with decay factor `beta`. """
    return (1 - beta**(k + 1)) / (1 - beta)

def optimize_for_max_gamma(betas, W_tile, C, max_gamma, precomputed_values):
    """ Optimize knapsack configuration for a specific max_gamma. """
    n = len(betas)
    dp = [0] * (W_tile + 1)
    gammas = [[0] * n for _ in range(W_tile + 1)]
    cost = C * max_gamma + 1

    for i, beta in enumerate(betas):
        for j in range(W_tile, 0, -1):
            for k in range(1, min(max_gamma + 1, j + 1)):
                new_value = dp[j - k] + (precomputed_values[i][k - 1] if k <= len(precomputed_values[i]) else compute_value(beta, k))
                if new_value > dp[j]:
                    dp[j] = new_value
                    gammas[j] = gammas[j - k].copy()
                    gammas[j][i] += k
                else:
                    # If no improvement in value is seen with this 'k', no point in trying higher 'k'
                    break

    return dp[W_tile] / cost, gammas[W_tile]

def find_optimal_draft_size(seq_group_list: List[SequenceGroup],
                            sps_config: SpSConfig):
    """ Explore optimal solutions by adjusting max_gamma up and down from the start point. """
    C = sps_config.target_draft_latency_ratio
    tile_constraint = sps_config.get_tile_size_constraint()
    start_max_draft_size = sps_config.start_max_draft_size
    betas = []

    for seq_group in seq_group_list:
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        betas.append(seq.get_beta())

    precomputed_values = [[compute_value(beta, k) for k in range(1, start_max_draft_size + 1)] for beta in betas]
    max_dp = float('-inf')
    result = None
    search_by_increasing = False

    # Decrease max_gamma to find the optimal point
    for max_draft_size in range(start_max_draft_size, 0, -1):
        current_dp, current_gammas = optimize_for_max_gamma(betas, tile_constraint, C, max_draft_size, precomputed_values)
        if current_dp > max_dp:
            max_dp = current_dp
            result = current_gammas
        else:
            # If second highest max_gamma did not improve the solution, we should search by increasing the max_gamma
            if max_draft_size != start_max_draft_size - 1:
                search_by_increasing = True
            break

    if search_by_increasing:
        # If decreasing didn't improve, try increasing
        increasing_gamma = start_max_draft_size + 1
        while True:
            current_dp, current_gammas = optimize_for_max_gamma(betas, tile_constraint, C, increasing_gamma, precomputed_values)
            if current_dp > max_dp:
                max_dp = current_dp
                result = current_gammas
                increasing_gamma += 1  # Continue to check higher max gamma
            else:
                break  # No improvement found, break the loop

    # Update draft size for each sequence group
    for i, seq_group in enumerate(seq_group_list):
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        seq.draft_size = result[i]
