#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the window size
window_size=5

python3 benchmark_sps_latency.py \
--dataset "$dataset" \
--num-prompts 1 \
--engine sps \
--draft-size "$window_size"

# python3 benchmark_sps_throughput.py \
# --dataset "$dataset" \
# --num-prompts 100 \
# --engine base \
# --draft-size "$window_size"
