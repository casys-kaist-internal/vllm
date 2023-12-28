#!/bin/bash

# get first args from bash script 
args=("$@")
engine=${args[0]}

if [ "$engine" != "base" ] && [ "$engine" != "sps" ]; then
    echo "Invalid engine: $engine"
    exit 1
fi

echo "Running $engine client"

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set target model 
target_model="facebook/opt-6.7b"

# Set the number of prompts
num_prompts=100

# Set request rate
request_rate=10

if [ "$engine" = "base" ]; then
    python3 benchmark_serving.py \
        --port 7777 \
        --tokenizer $target_model \
        --dataset $dataset \
        --num-prompts $num_prompts \
        --request-rate $request_rate
    exit 0
else
    python3 benchmark_serving.py \
        --port 8888 \
        --tokenizer $target_model \
        --dataset $dataset \
        --num-prompts $num_prompts \
        --request-rate $request_rate
    exit 0
fi
