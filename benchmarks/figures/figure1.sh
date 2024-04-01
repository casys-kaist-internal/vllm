#!/bin/bash
echo -n "$(pwd)"
base="/home/sjlim/workspace/vllm/benchmarks/figures"

for dataset in "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt"; do
    for (( batch_size=1; batch_size <= 128; batch_size*=2 ))
    do
        echo -n "$batch_size, " >> "${base}/warmpup/fig1_base_$dataset.csv"
        python3 figure1.py \
        --dataset "$dataset" \
        --engine base \
        --num-iters 10 \
        --batch-size "$batch_size" \
        | grep "result" >> "${base}/result/fig1_base_$dataset.csv"

        # Loop through draft size from 2 to 8
        for draft_size in {2..8}; do
            echo -n "$batch_size, $draft_size, " >> "${base}/result/fig1_sps_$dataset.csv"
            python3 figure1.py \
            --dataset "$dataset" \
            --engine sps \
            --num-iters 10 \
            --draft-size "$draft_size" \
            --batch-size "$batch_size" \
            | grep "result" >> "${base}/result/fig1_sps_$dataset.csv"
        done
        slack "Done $dataset $batch_size"
    done
done