#!/bin/bash

declare -a models=(
    # "facebook/opt-6.7b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-350m"
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-13b,facebook/opt-350m"
    # "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "bigscience/bloom-7b1,bigscience/bloomz-560m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-31m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-14m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)

datasets=("apps")
temperatures=(0.0)
batch_sizes=(1)

# Calculate total number of iterations
total_num_progress=$(( ${#models[@]} * ${#datasets[@]} * ${#temperatures[@]} * ${#batch_sizes[@]} ))
current_progress=0

for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    for batch_size in "${batch_sizes[@]}"; do
        for dataset in "${datasets[@]}"; do
            for temp in "${temperatures[@]}"; do 
                python3 5_17.py \
                    --target-model "$target_model" \
                    --draft-model "$draft_model" \
                    --dataset "$dataset" \
                    --temperature "$temp" \
                    --batch-size "$batch_size"
                
                # Increment current progress
                ((current_progress++))
                
                # Calculate and print progress percentage
                progress_percentage=$(awk "BEGIN {printf \"%.2f\", ($current_progress / $total_num_progress) * 100}")
                echo "Progress: $current_progress / $total_num_progress ($progress_percentage%)"
                ./slack "Progress: $current_progress / $total_num_progress ($progress_percentage%)"
            done
        done
    done
done
