#!/bin/bash

# Set the dataset path
dataset="/home/yhkim/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the output file path
output_file="/home/yhkim/workspace/vllm/benchmarks/results_latency.csv"

# Latency or throughput
benchmark="latency"

# List of models
opt_models="facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b"

# gpt2
gpt2_models="openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large openai-community/gpt2-xl"

# bloom
bloom_models="bigscience/bloom-560m bigscience/bloom-1b1 bigscience/bloom-1b7 bigscience/bloom-3b bigscience/bloom-7b1"

llama_models="openlm-research/open_llama_3b openlm-research/open_llama_7b" 

phi_models="microsoft/phi-1 microsoft/phi-1_5 microsoft/phi-2"

gpt_neo_models="EleutherAI/gpt-neo-125m EleutherAI/gpt-neo-1.3b EleutherAI/gpt-neo-2.7b"


# Loop through models
for model in $opt_models; do
    # Loop through batch sizes from 1 to 10
    for batch_size in {1..256}; do 
        output_file="/home/yhkim/workspace/vllm/benchmarks/result_context_len/${model}_${batch_size}.csv"
        if [ -f "$output_file" ]; then
            rm "$output_file"
        fi
        python3 benchmark_sps_$benchmark.py \
        --num-prompts 1 \
        --engine base \
        --input-len 1 \
        --output-len 2048 \
        --target-model "$model" \
        --batch-size "$batch_size" \
        | grep "latency" >> "$output_file"
        # {task}, {batch_size}, {context_len}, {latency(s)} 로 저장됨
        # tast = 'latency', context_len은 k번째 토큰을 생성하는 decoding step을 의미
        ## model init 반드시 확인할것
    done
done
