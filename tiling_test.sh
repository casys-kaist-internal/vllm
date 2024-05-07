#!/bin/bash

declare -a models=(
    "--target-model daryl149/llama-2-7b-chat-hf --draft-model Felladrin/Llama-68M-Chat-v1"
    "--target-model daryl149/llama-2-13b-chat-hf --draft-model Felladrin/Llama-68M-Chat-v1"
    "--target-model bigscience/bloom-7b1 --draft-model bigscience/bloomz-560m"
    "--target-model facebook/opt-13b --draft-model facebook/opt-125m"
    "--target-model facebook/opt-6.7b --draft-model facebook/opt-125m"
    "--target-model EleutherAI/pythia-12b --draft-model EleutherAI/pythia-14m"
    "--target-model EleutherAI/pythia-6.9b --draft-model EleutherAI/pythia-14m"
)

# Loop through each model
for model in "${models[@]}"; do
    python3 benchmarks/kernels/bench_layer_tiling.py --random-draft-len --random-context-len $model
done

