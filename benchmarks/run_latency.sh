#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/sjchoi/workspace/vllm/benchmarks/results_latency.csv"

# Latency or throughput
benchmark="latency"

# list of models
models=(

# Loop through batch sizes from 1 to 10
for batch_size in {1..100}; do 
    python3 benchmark_sps_$benchmark.py \
    --target-model lmsys/vicuna-13b-v1.3 \
    --dataset "$dataset" \
    --engine base \
    --batch-size "$batch_size" \
    | grep "latency" > "base_$batch_size.txt"
done
