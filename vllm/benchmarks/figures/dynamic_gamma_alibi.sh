#!/bin/bash

# Define the combinations of target model, draft model, and temperature
declare -a models=(
    "bigscience/bloom-7b1,bigscience/bloomz-560m"
)

# Define the temperatures
declare -a temperatures=(0.5)

# Define the datasets
declare -a datasets=("alpaca")

# Loop through each combination and run the script
for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    for temperature in "${temperatures[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Running with $target_model and $draft_model on dataset $dataset with temperature $temperature"
            python dynamic_gamma_alibi.py \
                --engine sps \
                --dataset "$dataset" \
                --target-model "$target_model" \
                --draft-model "$draft_model" \
                --temperature "$temperature" \
                --batch-size 1
        done
    done
done