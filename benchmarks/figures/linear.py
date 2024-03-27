import torch
import numpy as np
import time
import sys

# check number of arguments
if len(sys.argv) != 3:
    raise SystemExit(f"Usage: {sys.argv[0]} <input_features> <output_features>")

# Get input_features and output_features from command line arguments
input_features = int(sys.argv[1])
output_features = int(sys.argv[2])
num_iterations = 100

# Create a placeholder for profiling results and TFLOPS
profile_results = {}
tflops_results = {}
ai_results = {}

# Ensure CUDA is available and set the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using:", torch.cuda.get_device_name(device), "input_features:", input_features, "output_features:", output_features)
else:
    raise SystemExit("CUDA is not available. This script requires a GPU.")

# Loop over batch sizes from 1 to 512
for batch_size in range(1, 1025):
    # Define the input tensor and linear layer, moving them to the GPU
    input_tensor = torch.randn(batch_size, input_features, device=device).half()
    linear_layer = torch.nn.Linear(input_features, output_features).to(device).half()

    for _ in range(3):
        # Warm-up run for more accurate timing
        output = linear_layer(input_tensor)

    # Synchronize CUDA to ensure the GPU is ready
    torch.cuda.synchronize()
    # Start timing
    start_time = time.perf_counter_ns()

    for _ in range(num_iterations):
        # Perform the matrix multiplication
        output = linear_layer(input_tensor)

    # Synchronize CUDA to ensure completion of the operation
    torch.cuda.synchronize()
    # End timing
    end_time = time.perf_counter_ns()
    
    # Calculate and append the elapsed time to the list
    elapsed_time_ns = (end_time - start_time)
    elapsed_time_s = elapsed_time_ns / 1e9  # Convert to seconds
    avg_elapsed_time_s = elapsed_time_s / num_iterations
    
    # Calculate FLOPs per operation
    flops_per_operation = 2 * batch_size * input_features * output_features

    # Calculate MOPs per operation
    mops_per_operation = 2 * (batch_size * input_features + input_features * output_features + batch_size * output_features)

    # Calculate total FLOPs
    total_flops = flops_per_operation * num_iterations
    total_mops = mops_per_operation * num_iterations

    # Calculate TFLOPS
    tflops = (total_flops / elapsed_time_s) / 1e12

    # Calculate Arithmetic Intensity
    ai = total_flops / total_mops

    profile_results[batch_size] = avg_elapsed_time_s * 1e3  # Convert to milliseconds for readability
    tflops_results[batch_size] = tflops
    ai_results[batch_size] = ai

    print(f"Batch Size: {batch_size}, Average Time Taken: {profile_results[batch_size]:.3f} ms, TFLOPS: {tflops_results[batch_size]:.3f}, AI: {ai_results[batch_size]:.3f}")

# At this point, `profile_results` and `tflops_results` contain the average profiling information and TFLOPS for each batch size.
# save csv file
import csv
import sys
with open(f"{torch.cuda.get_device_name(device)}_{input_features}_{output_features}.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Batch Size', 'Average Time (ms)', 'TFLOPS', 'AI'])
    for batch_size in profile_results:
        writer.writerow([batch_size, profile_results[batch_size], tflops_results[batch_size], ai_results[batch_size]])
