#!/bin/bash
echo -n "$(pwd)"
base="/home/sjlim/workspace/vllm/benchmarks/figures"
temperatures=(0 0.3 0.5 0.7 1)

for dataset in "gsm8k" "humaneval" "alpaca" "mt-bench" "sharegpt"; do
    for temperature in "${temperatures[@]}"; do
        for (( batch_size=64; batch_size <= 64; batch_size*=2 ))
        do
            echo -n "$batch_size, $temperature," >> "${base}/result/out_base_$dataset.csv"

            python3 output.py \
            --dataset "$dataset" \
            --engine base \
            --num-iters 1 \
            --batch-size "$batch_size" \
            --temperature "$temperature" \
            | grep "result" >> "${base}/result/out_base_$dataset.csv"

            # Loop through draft size from 2 to 8
            for draft_size in {2..8}; do
                echo -n "$batch_size, $temperature, $draft_size, " >> "${base}/result/out_sps_$dataset.csv"
                python3 output.py \
                --dataset "$dataset" \
                --engine sps \
                --num-iters 1 \
                --draft-size "$draft_size" \
                --batch-size "$batch_size" \
                --temperature "$temperature" \
                | grep "result" >> "${base}/result/out_sps_$dataset.csv"
            done
            slack "Done $dataset $batch_size"
        done
    done
done