#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/sjchoi/workspace/vllm/benchmarks/results_latency.csv"

# Set the number of prompts
num_prompts=100

# Latency or throughput
benchmark="latency"

rm $output_file

# Loop through batch sizes from 1 to 10
for batch_size in {2..10}; do 
    echo -n "Base, $batch_size," >> "$output_file"
    python3 benchmark_sps_$benchmark.py \
    --dataset "$dataset" \
    --num-prompts "$num_prompts" \
    --engine base \
    --batch-size "$batch_size" \
    | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"

    # Loop through window sizes from 2 to 10
    for window_size in {2..8}; do
        # Run the benchmark with sps engine and current window size
        echo -n "SpS_$window_size, $batch_size," >> "$output_file"
        python3 benchmark_sps_$benchmark.py \
        --dataset "$dataset" \
        --num-prompts "$num_prompts" \
        --engine sps \
        --draft-size "$window_size" \
        --batch-size "$batch_size" \
        | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"
    done
done
