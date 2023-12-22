#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/sjchoi/workspace/vllm/benchmarks/results.csv"

python3 benchmark_sps_latency.py --dataset "$dataset" --engine base >> "$output_file"

# Loop through window sizes from 2 to 8
for window_size in {2..8}; do
    # Run the benchmark with sps engine and current window size
    python3 benchmark_sps_latency.py --dataset "$dataset" --engine sps --draft-size "$window_size" >> "$output_file"
done
