#!/bin/bash

declare -a models=(
    # "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "bigscience/bloom-7b1,bigscience/bloomz-560m"
    # "facebook/opt-13b,facebook/opt-350m"
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-350m"
    "facebook/opt-6.7b,facebook/opt-125m"
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

device=0
export CUDA_VISIBLE_DEVICES=$device

dir_name="result_4_30"
# Create result directory
rm -rf $dir_name
mkdir -p $dir_name
batch_size=1

for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    for index in {1..10}; do
        for dataset in "apps"; do
            # Loop through draft size from 2 to 7
            for draft_size in {2..7}; do
                echo -n "$target_model, $draft_model, $batch_size, $draft_size, " >> "$dir_name/fig1_sps_$dataset.csv"
                python3 4_30.py \
                --target-model "$target_model" \
                --draft-model "$draft_model" \
                --dataset "$dataset" \
                --engine sps \
                --num-iters 1 \
                --index "$index" \
                --draft-size "$draft_size" \
                --batch-size "$batch_size" \
                | grep "latency" >> "$dir_name/fig1_sps_$dataset.csv"
            done

            echo -n "$target_model, $draft_model, $batch_size, dynamic, " >> "$dir_name/fig1_sps_$dataset.csv"
            python3 4_30.py \
                --target-model "$target_model" \
                --draft-model "$draft_model" \
                --dataset "$dataset" \
                --engine sps \
                --num-iters 1 \
                --index "$index" \
                --batch-size "$batch_size" \
                --dynamic-draft \
                | grep "latency" >> "$dir_name/fig1_sps_$dataset.csv"

            # # Loop through draft size from 2 to 8
            # for draft_size in {2..7}; do
            #     echo -n "$target_model, $draft_model, $batch_size, $draft_size, " >> "$dir_name/fig1_sps_target_$dataset.csv"
            #     python3 figure1.py \
            #     --target-model "$target_model" \
            #     --draft-model "$draft_model" \
            #     --dataset "$dataset" \
            #     --engine sps \
            #     --num-iters 1 \
            #     --draft-size "$draft_size" \
            #     --batch-size "$batch_size" \
            #     --use-target-attention \
            #     | grep "latency" >> "$dir_name/fig1_sps_target_$dataset.csv"
            # done

            slack "Done $dataset $batch_size"
        done
    done
done
