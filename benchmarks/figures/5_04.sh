#!/bin/bash

declare -a models=(
    "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    "bigscience/bloom-7b1,bigscience/bloomz-560m"
    "facebook/opt-13b,facebook/opt-350m"
    "facebook/opt-13b,facebook/opt-125m"
    "facebook/opt-6.7b,facebook/opt-350m"
    "facebook/opt-6.7b,facebook/opt-125m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-410m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-160m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-70m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-31m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-14m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-410m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-70m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)

device=0
export CUDA_VISIBLE_DEVICES=$device

dir_name="5_04_result"
# log_file="$dir_name/error.log"
log_file="/dev/null"

# Create result directory
rm -rf $dir_name
mkdir -p $dir_name

# Constant parameters
batch_size=16
draft_size=8

counter=0

for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    # for dataset in "sharegpt"; do
    for dataset in "apps" "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt"; do
        # for temp in 0.5; do 
        for temp in 0.0 0.25 0.5 0.75 1.0; do 
            counter=$((counter+1))
            fname="$dir_name/$counter.csv"
            echo "$dataset, $target_model, $draft_model, $temp" > "$fname"
            python3 5_04.py \
                --target-model "$target_model" \
                --draft-model "$draft_model" \
                --dataset "$dataset" \
                --temperature "$temp" \
                --engine sps \
                --draft-size "$draft_size" \
                --batch-size "$batch_size" \
                2>> "$log_file" \
                | grep "accept" >> "$fname"
            { python process_beta_list_log.py -f "$fname" && rm "$fname"; } & # Clean up in the background.
        done
    done
    echo "Done $target_model $draft_model"
done

wait
