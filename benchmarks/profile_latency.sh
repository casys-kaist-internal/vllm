#!/bin/bash
declare -a models=(
    "daryl149/llama-2-7b-chat-hf"
    "Felladrin/Llama-68M-Chat-v1"
    "bigscience/bloom-7b1"
    "bigscience/bloomz-560m"
    "facebook/opt-13b"
    "facebook/opt-6.7b"
    "facebook/opt-1.3b"
    "facebook/opt-350m"
    "facebook/opt-125m"
    "EleutherAI/pythia-12b"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-410m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-31m"
    "EleutherAI/pythia-14m"
)

# Loop through each model 
for model in "${models[@]}"; do
    echo "Running with $model"
    # Get model name from $model with split by "/" and the last one 
    model_name=$(echo "$model" | tr "/" "\n" | tail -n 1)
    # Create output file
    output_file="output_latency_A100/$model_name.csv"
    for batch_size in 1 2 4 8 16 32 64; do
        # Get output and grep "profile keyword"
        output=$(python benchmark_sps_latency.py \
            --target-model "$model" \
            --batch-size "$batch_size" \
            | grep "profile,")
        echo "$output" >> $output_file
    done
done