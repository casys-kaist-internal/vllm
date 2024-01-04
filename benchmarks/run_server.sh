#!/bin/bash

# get first args from bash script 
args=("$@")
engine=${args[0]}
port=${args[1]}
draft_size=${args[2]}

if [ "$engine" != "base" ] && [ "$engine" != "sps" ]; then
    echo "Invalid engine: $engine"
    exit 1
fi

# check that port is a valid integer of length 4
if ! [[ "$port" =~ ^[0-9]+$ ]] || [ ${#port} -ne 4 ]; then
    echo "Invalid port: $port"
    exit 1
fi

download_dir="/home/sjchoi/workspace/models"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
OUTFILE='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')

# if engine is base, run base server. 
if [ "$engine" = "base" ]; then
    echo "Running $engine server at port $port"
    nsys profile --gpu-metrics-device=0 --output=${OUTFILE} \
    python3 -m vllm.entrypoints.api_server \
        --port $port \
        --download-dir $download_dir \
        --model $target_model \
        --disable-log-requests
    exit 0
else
    # check that draft_size is valid integer 
    if ! [[ "$draft_size" =~ ^[0-9]+$ ]]; then
        echo "Invalid draft_size: $draft_size"
        exit 1
    fi

    echo "Running $engine server at port $port with draft_size $draft_size"

    nsys profile --gpu-metrics-device=0 --output=${OUTFILE} \
    python3 -m vllm.entrypoints.sps_api_server \
        --port $port \
        --download-dir $download_dir \
        --target-model $target_model \
        --draft-model $draft_model \
        --draft-size $draft_size \
        --disable-log-requests
    exit 0
fi