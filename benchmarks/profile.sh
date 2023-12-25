#!/bin/bash

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the window size
window_size=4

OUTFILE='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')
# nsys profile --gpu-metrics-device=all --output=${OUTFILE} \
# python3 benchmark_sps_latency.py \
# --dataset "$dataset" \
# --num-prompts 1 \
# --engine sps \
# --draft-size "$window_size" \
# --output-len 100

nsys profile --gpu-metrics-device=0 --output=${OUTFILE} \
python3 benchmark_sps_throughput.py \
--dataset "$dataset" \
--num-prompts 100 \
--engine sps \
--draft-size "$window_size" \
--output-len 100