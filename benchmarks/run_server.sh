#!/bin/bash

# get first args from bash script 
args=("$@")
engine=${args[0]}

if [ "$engine" != "base" ] && [ "$engine" != "sps" ]; then
    echo "Invalid engine: $engine"
    exit 1
fi

echo "Running $engine server"

download_dir="/home/sjchoi/workspace/models"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=2

# if engine is base, run base server. 
if [ "$engine" = "base" ]; then
    python3 -m vllm.entrypoints.api_server \
        --port 7777 \
        --download-dir $download_dir \
        --model $target_model \
        --disable-log-requests
    exit 0
else
    python3 -m vllm.entrypoints.sps_api_server \
        --port 8888 \
        --download-dir $download_dir \
        --target-model $target_model \
        --draft-model $draft_model \
        --draft-size $draft_size \
        --disable-log-requests
    exit 0
fi