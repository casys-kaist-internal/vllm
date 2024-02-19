#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/sjchoi/workspace/vllm/benchmarks/results_latency.csv"

# Latency or throughput
benchmark="latency"

# List of models
models="facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b"

# Loop through models
for model in $models; do
    # Loop through batch sizes from 1 to 10
    for batch_size in {1..256}; do 
        output_file="/home/sjchoi/workspace/vllm/benchmarks/results/${model}_${batch_size}.csv"
        rm "$output_file"
        python3 benchmark_sps_$benchmark.py \
        --dataset "$dataset" \
        --num-prompts 1 \
        --engine base \
        --target-model "$model" \
        --batch-size "$batch_size" \
        | grep "latency" >> "$output_file"
    done
done