#!/bin/bash

declare -a models=(
    # "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "bigscience/bloom-7b1,bigscience/bloomz-560m"
    # "facebook/opt-13b,facebook/opt-350m"
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-350m"
    "facebook/opt-6.7b,facebook/opt-125m"
    "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-14m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)

device=1
export CUDA_VISIBLE_DEVICES=$device

for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    for temp in 0.75 0.5 1.0 0.25 0.0; do
        for dataset in "mt-bench" "humaneval" "apps" "gsm8k" "alpaca" "sharegpt"; do
            python3 log_reject_pos.py --target-model "$target_model" \
                        --draft-model "$draft_model" \
                        --dataset "$dataset" --temperature $temp
        done
    done
done
