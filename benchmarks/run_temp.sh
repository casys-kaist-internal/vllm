#!/bin/bash

# Set the dataset path
dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/noppanat/workspace/vllm/benchmarks/results.csv"

# Set the number of prompts
num_prompts=10

rm $output_file

# Loop through output_len from 128 256 512 1024 2048
for output_len in 128 256 512 1024 2048; do
    # Loop through temperature range
    for temperature in $(seq 0 0.1 1); do
        latency=$(python3 benchmark_sps_latency.py \
        --dataset "$dataset" \
        --num-prompts "$num_prompts" \
        --engine base \
        --temperature "$temperature" \
        --output-len "$output_len" \
        | grep "latency" | awk '{printf "%.3f\n", $3}')

        echo -n "$output_len, $temperature, Base, " >> "$output_file"
        echo $latency >> "$output_file"
        slack "$output_len, $temperature, Base, $latency"

        # Loop through window sizes from 2 to 10
        for window_size in {2..10}; do
            # Run the benchmark with sps engine, current window size, and temperature
            latency=$(python3 benchmark_sps_latency.py \
            --dataset "$dataset" \
            --num-prompts "$num_prompts" \
            --engine sps \
            --draft-size "$window_size" \
            --temperature "$temperature" \
            --output-len "$output_len" \
            | grep "latency" | awk '{printf "%.3f\n", $3}')

            echo -n "$output_len, $temperature, SpS $window_size, " >> "$output_file"
            echo $latency >> "$output_file"
            slack "$output_len, $temperature, SpS $window_size, $latency"
        done
    done
done

slack "Done"