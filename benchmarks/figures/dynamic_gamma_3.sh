#!/bin/bash

# Define the combinations of target model, draft model, and temperature
declare -a models=(
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)
# Define the temperatures
declare -a temperatures=(0.0 0.5 1.0)

# Define the datasets
declare -a datasets=("apps" "alpaca" "sharegpt" "gsm8k")

total_runs=$(( ${#models[@]} * ${#temperatures[@]} * ${#datasets[@]} ))
current_run=0
# Loop through each combination and run the script
for temperature in "${temperatures[@]}"; do
    for model_pair in "${models[@]}"; do
        IFS=',' read -r target_model draft_model <<< "$model_pair"
        for dataset in "${datasets[@]}"; do
            echo "Running with $target_model and $draft_model on dataset $dataset with temperature $temperature"
            ./slack "Running with $target_model and $draft_model on dataset $dataset with temperature $temperature"
            python dynamic_gamma.py \
                --engine sps \
                --dataset "$dataset" \
                --target-model "$target_model" \
                --draft-model "$draft_model" \
                --temperature "$temperature" \
                --batch-size 8
            current_run=$((current_run + 1))
            ./slack "[3] ${current_run}/${total_runs}"
        done
    done
done