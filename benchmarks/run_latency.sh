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
for batch_size in {1..16}; do 
    echo -n "Base, $batch_size," >> "$output_file"
    python3 benchmark_sps_$benchmark.py \
    --dataset "$dataset" \
    --num-prompts "$num_prompts" \
    --engine base \
    --batch-size "$batch_size" \
    | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"

    # Run the benchmark with sps engine and current window size
    echo -n "SpS, $batch_size," >> "$output_file"
    python3 benchmark_sps_$benchmark.py \
    --dataset "$dataset" \
    --num-prompts "$num_prompts" \
    --engine sps \
    --batch-size "$batch_size" \
    | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"
done
