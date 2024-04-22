#!/bin/bash
echo -n "$(pwd)"
base="/home/sjlim/workspace/vllm/benchmarks/figures"

for dataset in "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt"; do
    for (( batch_size=1; batch_size <= 1; batch_size*=2 ))
    do
        # Loop through draft size from 2 to 8
        for draft_size in {2..8}; do
            python3 figure1.py \
            --dataset "$dataset" \
            --engine sps \
            --num-iters 1 \
            --draft-size "$draft_size" \
            --batch-size "$batch_size" \
            | grep "result" >> "${base}/alpha/alpha_sps_${dataset}_${draft_size}_${batch_size}.csv"
        done
        slack "Done $dataset $batch_size"
    done
done