#!/bin/bash

# get first args from bash script 
args=("$@")
port=${args[0]}
request_rate=${args[1]}

# check that port is a valid integer of length 4
if ! [[ "$port" =~ ^[0-9]+$ ]] || [ ${#port} -ne 4 ]; then
    echo "Invalid port: $port"
    exit 1
fi

# check that request rate is a valid integer or is "inf"
if ! [[ "$request_rate" =~ ^[0-9]+$ ]] && [ "$request_rate" != "inf" ]; then
    echo "Invalid request rate: $request_rate"
    exit 1
fi

echo "Running client at port $port with $request_rate reqs/sec"

# Set the dataset path
dataset="/home/sjchoi/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set target model 
target_model="facebook/opt-6.7b"

# Set the number of prompts
num_prompts=100

python3 benchmark_serving.py \
    --port $port \
    --tokenizer $target_model \
    --dataset $dataset \
    --num-prompts $num_prompts \
    --request-rate $request_rate
exit 0
