#!/bin/bash

declare -a models=(
    "facebook/opt-6.7b,facebook/opt-125m"
    "facebook/opt-6.7b,facebook/opt-350m"
    "facebook/opt-13b,facebook/opt-125m"
    "facebook/opt-13b,facebook/opt-350m"
    "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    "bigscience/bloom-7b1,bigscience/bloomz-560m"
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

datasets=("apps" "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt" "all")
temperatures=(0.0 0.25 0.5 0.75 1.0)
batch_sizes=(32)
predictor_degree=(2 3)
predictor_agg_type=("median" "mean")

# Calculate total number of iterations
total_num_progress=$(( ${#models[@]} * ${#datasets[@]} * ${#temperatures[@]} * ${#batch_sizes[@]} * ${#predictor_degree[@]} * ${#predictor_agg_type[@]}))
current_progress=0


for predictor_agg_type in "${predictor_agg_type[@]}"; do
    for predictor_degree in "${predictor_degree[@]}"; do
        for model_pair in "${models[@]}"; do
            IFS=',' read -r target_model draft_model <<< "$model_pair"
            for batch_size in "${batch_sizes[@]}"; do
                for dataset in "${datasets[@]}"; do
                    for temp in "${temperatures[@]}"; do 
                        python3 5_20.py \
                            --target-model "$target_model" \
                            --draft-model "$draft_model" \
                            --dataset "$dataset" \
                            --temperature "$temp" \
                            --batch-size "$batch_size" \
                            --predictor-degree "$predictor_degree" \
                            --predictor-agg-type "$predictor_agg_type"
                        
                        # Increment current progress
                        ((current_progress++))
                        
                        # Calculate and print progress percentage
                        progress_percentage=$(echo "scale=2; ($current_progress / $total_num_progress) * 100" | bc)
                        echo "Progress: $current_progress / $total_num_progress ($progress_percentage%)"
                        ./slack "[A100] Progress: $current_progress / $total_num_progress ($progress_percentage%)"
                    done
                done
            done
        done
    done
done
