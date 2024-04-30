from typing import List
from vllm.config import SpSConfig
from vllm.sequence import (SequenceGroup, SequenceStatus)

def objective(gammas, betas, C):
    max_gamma = max(gammas)
    assert max_gamma > 0

    sum_terms = 0
    for gamma, beta in zip(gammas, betas):
        assert beta != 1
        term = (1 - beta**(gamma + 1)) / (1 - beta)
        sum_terms += term

    objective_value = sum_terms / (C * max_gamma + 1)
    return objective_value

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

def find_optimal_draft_size_with_tile_constraint(seq_group_list: List[SequenceGroup],
                            sps_config: SpSConfig):
    """ Explore optimal solutions by adjusting max_gamma up and down from the start point. """
    C = sps_config.target_draft_latency_ratio
    # tile_constraint = sps_config.get_tile_size_constraint()
    tile_constraint = 54
    start_max_draft_size = sps_config.start_max_draft_size
    betas = []

    for seq_group in seq_group_list:
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        betas.append(seq.get_beta())

    # print("Betas", betas)

    precomputed_values = [[compute_value(beta, k) for k in range(1, start_max_draft_size + 1)] for beta in betas]
    max_dp = float('-inf')
    result = None
    search_by_increasing = False

    # Decrease max_gamma to find the optimal point
    for max_draft_size in range(start_max_draft_size, 0, -1):
        current_dp, current_draft_sizes = optimize_for_max_gamma(betas, tile_constraint, C, max_draft_size, precomputed_values)
        if current_dp > max_dp:
            max_dp = current_dp
            result = current_draft_sizes
        else:
            # If second highest max_gamma did not improve the solution, we should search by increasing the max_gamma
            if max_draft_size != start_max_draft_size - 1:
                search_by_increasing = True
            break

    if search_by_increasing:
        # If decreasing didn't improve, try increasing
        increasing_draft_size = start_max_draft_size + 1
        while True:
            current_dp, current_draft_sizes = optimize_for_max_gamma(betas, tile_constraint, C, increasing_draft_size, precomputed_values)
            if current_dp > max_dp:
                max_dp = current_dp
                result = current_draft_sizes
                increasing_draft_size += 1  # Continue to check higher max gamma
            else:
                break  # No improvement found, break the loop

    # Update draft size for each sequence group
    for i, seq_group in enumerate(seq_group_list):
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        seq.draft_size = result[i]

def find_optimal_draft_size_without_tile_constraint(seq_group_list: List[SequenceGroup],
                                                    sps_config: SpSConfig):
    for i, seq_group in enumerate(seq_group_list):
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        # round up to the nearest int value
        seq.draft_size = int(seq.get_beta_ema())

        if seq.draft_size < 1:
            seq.draft_size = 1
        elif seq.draft_size > 7:
            seq.draft_size = 7

    # """ Find the optimal draft size without tile constraint. """
    # C = sps_config.target_draft_latency_ratio
    # start_max_draft_size = sps_config.start_max_draft_size
    # betas = []

    # for seq_group in seq_group_list:
    #     seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
    #     betas.append(seq.get_beta_ema())

    # max_value = float('-inf')
    # result = None
    # search_by_increasing = False

    # # Decrease max_gamma to find the optimal point
    # for max_draft_size in range(start_max_draft_size, 0, -1):
    #     current_value = objective([max_draft_size] * len(betas), betas, C)
    #     if current_value > max_value:
    #         max_value = current_value
    #         result = [max_draft_size] * len(betas)
    #     else:
    #         # If second highest max_gamma did not improve the solution, we should search by increasing the max_gamma
    #         if max_draft_size != start_max_draft_size - 1:
    #             search_by_increasing = True
    #         break

    # # if search_by_increasing:
    # #     # If decreasing didn't improve, try increasing
    # #     increasing_draft_size = start_max_draft_size + 1
    # #     while True:
    # #         current_value = objective([increasing_draft_size] * len(betas), betas, C)
    # #         if current_value > max_value:
    # #             max_value = current_value
    # #             result = [increasing_draft_size] * len(betas)
    # #             increasing_draft_size += 1
    # #         else:
    # #             break

    # assert result is not None

    # # print(result[0])
    # # Update draft size for each sequence group
    # for i, seq_group in enumerate(seq_group_list):
    #     seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
    #     seq.draft_size = result[i]