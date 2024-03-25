import torch
import numpy as np
import time

# Define the dimensions for the matrix multiplication
input_features = 4096
output_features = 4096
num_iterations = 100

# Create a placeholder for profiling results
profile_results = {}

# Ensure CUDA is available and set the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using:", torch.cuda.get_device_name(device))
else:
    raise SystemExit("CUDA is not available. This script requires a GPU.")

# Loop over batch sizes from 1 to 256
for batch_size in range(1, 513):
        # Define the input tensor and linear layer, moving them to the GPU
        input_tensor = torch.randn(batch_size, input_features, device=device)
        linear_layer = torch.nn.Linear(input_features, output_features).to(device)

        for _ in range(3):
            # Warm-up run for more accurate timing
            _ = linear_layer(input_tensor)

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
        avg_latency = (end_time - start_time) / num_iterations
        avg_latency = avg_latency / 1e6  # Convert to milliseconds
        
        profile_results[batch_size] = avg_latency

        print(f"Batch Size: {batch_size}, Average Time Taken: {avg_latency:.3f} ms")

# At this point, `profile_results` contains the average profiling information for each batch size
# save it to csv file 
import csv
with open('linear_profiling_results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Batch Size', 'Average Latency (ms)'])
    for key, value in profile_results.items():
        writer.writerow([key, value])