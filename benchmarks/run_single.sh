#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the window size
window_size=2

./nsys_profile python3 benchmark_sps_latency.py \
--dataset "$dataset" \
--num-prompts 24 \
--batch-size 24 \
--engine base \
--draft-size "$window_size"

# python3 benchmark_sps_throughput.py \
# --dataset "$dataset" \
# --num-prompts 100 \
# --engine base \
# --draft-size "$window_size"
