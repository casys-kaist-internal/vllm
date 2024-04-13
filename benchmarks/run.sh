#!/bin/bash

# Set the dataset path
dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/noppanat/workspace/vllm/benchmarks/results_batch_size.csv"

# Set the number of prompts
total_run=10

rm $output_file

# Loop through batch sizes from 1 to 10
for batch_size in {1..64}; do
    num_prompts=$((batch_size * total_run))
    echo -n "Base, $batch_size, " >> "$output_file"
    python3 benchmark_sps_latency.py \
    --dataset "$dataset" \
    --num-prompts "$num_prompts" \
    --engine base \
    --batch-size "$batch_size" \
    | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"

    # Loop through window sizes from 2 to 10
    for window_size in {2..10}; do
        # Run the benchmark with sps engine and current window size
        echo -n "SpS $window_size, $batch_size, " >> "$output_file"
        python3 benchmark_sps_latency.py \
        --dataset "$dataset" \
        --num-prompts "$num_prompts" \
        --engine sps \
        --batch-size "$batch_size" \
        --draft-size "$window_size" \
        | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"
    done
done