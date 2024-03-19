#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/sjchoi/workspace/vllm/benchmarks/results.csv"

# Set the number of prompts
num_prompts=100

# Latency or throughput
benchmark="throughput"

rm $output_file

echo "Engine, requests/s, tokens/s" >> "$output_file"

# num_prompts=11
# ./nsys_profile python3 benchmark_sps_$benchmark.py \
#     --dataset "$dataset" \
#     --num-prompts "$num_prompts" \
#     --engine base

# num_prompts=11
# window_size=4
# ./nsys_profile python3 benchmark_sps_$benchmark.py \
#     --dataset "$dataset" \
#     --num-prompts "$num_prompts" \
#     --engine sps \
#     --draft-size "$window_size"

# Loop through num_prompts from 1 to 100 with step 10
for num_prompts in {1..20}; do
    echo -n "$num_prompts, Base, " >> "$output_file"
    python3 benchmark_sps_$benchmark.py \
    --dataset "$dataset" \
    --num-prompts "$num_prompts" \
    --engine base \
    | grep "Throughput" | awk -F'[ ,]' '{print $2", "$5}' >> "$output_file"

    # Loop through window sizes from 2 to 5
    for window_size in {2..5}; do
        # Run the benchmark with sps engine and current window size
        echo -n "$num_prompts, SpS $window_size, " >> "$output_file"
        python3 benchmark_sps_$benchmark.py \
        --dataset "$dataset" \
        --num-prompts "$num_prompts" \
        --engine sps \
        --draft-size "$window_size" \
        | grep "Throughput" | awk -F'[ ,]' '{print $2", "$5}' >> "$output_file"
    done
done