import random
import time 
from itertools import product

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

def brute_force(betas, W_tile, C):
    n = len(betas)
    best_value = float('-inf')
    best_gammas = None

    # Generate all combinations of gammas that sum to less than or equal to W_tile
    for gammas in product(range(W_tile + 1), repeat=n):
        # count number of non-zeros in gamma 
        if sum(gammas) <= W_tile and sum(gammas) > 0:
            current_value = objective(gammas, betas, C)
            # print(gammas, current_value)
            if current_value > best_value:
                best_value = current_value
                best_gammas = gammas
    
    return best_gammas, best_value

def dp(betas, W_tile, C, end_max_gamma):
    n = len(betas)
    # precomputed values [n, max_gamma_limit]
    precomputed_values = [[(1 - betas[i]**(k + 1)) / (1 - betas[i]) for k in range(1, end_max_gamma + 1)] for i in range(n)]
    max_dp = float('-inf')
    max_gammas = None

    for max_gamma in range(end_max_gamma, 0, -1):
        dp = [0] * (W_tile + 1)
        gammas = [[0] * n for _ in range(W_tile + 1)]
        cost = C * max_gamma + 1

        for i in range(n):
            for j in range(W_tile, 0, -1):
                for k in range(1, min(max_gamma + 1, j + 1)):
                    new_value = dp[j - k] + precomputed_values[i][k - 1]
                    if new_value > dp[j]:
                        dp[j] = new_value
                        gammas[j] = gammas[j - k].copy()
                        gammas[j][i] = gammas[j - k][i] + k
                    else:
                        # If no improvement in value is seen with this 'k', no point in trying higher 'k'
                        break
        
        if dp[W_tile] / cost > max_dp:
            max_dp = dp[W_tile] / cost
            max_gammas = gammas[W_tile]
        else:
            # No improvement in value, no point in trying lower max_gamma
            # But if this happend in the second highest max_gamma, there is possibility that we can get better value with higher max_gamma
            if max_gamma == end_max_gamma - 1:
                max_gamma = end_max_gamma
                while True: 
                    max_gamma += 1
                    dp = [0] * (W_tile + 1)
                    gammas = [[0] * n for _ in range(W_tile + 1)]
                    cost = C * max_gamma + 1

                    for i in range(n):
                        for j in range(W_tile, 0, -1):
                            for k in range(1, min(max_gamma + 1, j + 1)):
                                # Since there is no precomputed value, we need to calculate it on the fly
                                if k > end_max_gamma:
                                    new_value = dp[j - k] + (1 - betas[i]**(k + 1)) / (1 - betas[i])
                                else:
                                    new_value = dp[j - k] + precomputed_values[i][k - 1]

                                if new_value > dp[j]:
                                    dp[j] = new_value
                                    gammas[j] = gammas[j - k].copy()
                                    gammas[j][i] = gammas[j - k][i] + k
                                else:
                                    # If no improvement in value is seen with this 'k', no point in trying higher 'k'
                                    break

                    if dp[W_tile] / cost > max_dp:
                        max_dp = dp[W_tile] / cost
                        max_gammas = gammas[W_tile]
                        # Keep trying higher max_gamma
                    else:
                        # No improvement in value, break
                        break
            break
    
    return max_dp, max_gammas

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

def find_optimal_max_gamma(betas, W_tile, C, start_max_gamma):
    """ Explore optimal solutions by adjusting max_gamma up and down from the start point. """
    precomputed_values = [[compute_value(beta, k) for k in range(1, start_max_gamma + 1)] for beta in betas]
    max_dp = float('-inf')
    max_gammas = None

    # Decrease max_gamma to find the optimal point
    for max_gamma in range(start_max_gamma, 0, -1):
        current_dp, current_gammas = optimize_for_max_gamma(betas, W_tile, C, max_gamma, precomputed_values)
        if current_dp > max_dp:
            max_dp = current_dp
            max_gammas = current_gammas
        else:
            # If second highest max_gamma did not improve the solution, we should search by increasing the max_gamma
            if max_gamma != start_max_gamma - 1:
                return max_dp, max_gammas   # No improvement found, return the current best solution
            break

    # If decreasing didn't improve, try increasing
    increasing_gamma = start_max_gamma + 1
    while True:
        current_dp, current_gammas = optimize_for_max_gamma(betas, W_tile, C, increasing_gamma, precomputed_values)
        if current_dp > max_dp:
            max_dp = current_dp
            max_gammas = current_gammas
            increasing_gamma += 1  # Continue to check higher max gamma
        else:
            break  # No improvement found, break the loop

    return max_dp, max_gammas

# def binary_search_max_gamma(betas, W_tile, C, low, high):
#     # Precompute values for potential max_gamma
#     max_precomputed_gamma = high
#     precomputed_values = [[compute_additional_value(beta, k) for k in range(1, max_precomputed_gamma + 1)] for beta in betas]

#     best_value = float('-inf')
#     best_gamma = None

#     while low <= high:
#         mid = (low + high) // 2
#         current_value = optimize_for_max_gamma(betas, W_tile, C, mid, precomputed_values)

#         # To evaluate the neighbors
#         left_value = optimize_for_max_gamma(betas, W_tile, C, mid - 1, precomputed_values) if mid > low else float('-inf')
#         right_value = optimize_for_max_gamma(betas, W_tile, C, mid + 1, precomputed_values) if mid < high else float('-inf')

#         if current_value > left_value and current_value > right_value:
#             best_value = current_value
#             best_gamma = mid
#             break  # Found the optimal point
#         elif left_value > current_value:
#             high = mid - 1
#         else:
#             low = mid + 1

#     return best_value, best_gamma

# def objective(gammas, betas, C):
#     max_gamma = max(gammas)
#     assert max_gamma > 0

#     sum_terms = 0
#     for gamma, beta in zip(gammas, betas):
#         assert beta != 1
#         term = (1 - beta**(gamma + 1)) / (1 - beta)
#         sum_terms += term

#     objective_value = sum_terms / (C * max_gamma + 1)
#     return objective_value


def main():
    # Example parameters
    # betas = []
    n = 32
    betas = [random.uniform(0.5, 0.9) for _ in range(n)]
    W_tile = 128  # Total weight constraint
    C = 0.03  # Constant C, could be adjusted based on the problem specifics

    print("Number of requests:", n)
    print("Betas:", betas)
    print("Tile size constraint:", W_tile)

    # # # Brute force approach to find the optimal gamma values
    # start_time = time.time()
    # brute_force_result = brute_force(betas, W_tile, C)
    # end_time = time.time()
    # print("Brute Force: Time taken:", end_time - start_time)
    # print("Gamma values:", brute_force_result)

     # Call the DP algorithm
    # start_time = time.time()
    # result = dp(betas, W_tile, C, end_max_gamma=8)
    # end_time = time.time()
    # print("Dynamic Programming: Time taken:", end_time - start_time)
    # # Find max value from dp_results 
    # # Get the index of the max value 
    # print(f"{result[0]}, {result[1]}")

    start_time = time.time()
    result = find_optimal_max_gamma(betas, W_tile, C, start_max_gamma=8)
    end_time = time.time()
    print("Find Optimal Max Gamma: Time taken:", end_time - start_time)
    print(f"{result[0]}, {result[1]}")

    # # Call the binary search algorithm
    # start_time = time.time()
    # binary_search_result = binary_search_max_gamma(betas, W_tile, C, low=1, high=16)
    # end_time = time.time()
    # print("Binary Search: Time taken:", end_time - start_time)
    # print(f"{binary_search_result[0]}, {binary_search_result[1]}")

    # Correctness between brute force and DP
    # if list(brute_force_result[0]) != result[1]:
    #     print("Failed!")

if __name__ == "__main__":
    main()
