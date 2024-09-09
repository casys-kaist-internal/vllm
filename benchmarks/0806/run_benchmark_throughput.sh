#!/bin/bash

mkdir -p batch_size

# Loop through batch sizes from 1 to 128
for batch_size in {1..128}; do
    # Execute the Python script with the current batch size
    python benchmark_throughput.py --temperature 0 --input-len 32 --output-len 128 --batch-size "$batch_size" --draft-size 7 --prefill-schedule-mode prioritize_prefill --budget-token 2048 --budget-seq 256 --drop-threshold 0 --target-model facebook/opt-6.7b --draft-model facebook/opt-125m > output.log
    
    # Extract lines containing "Budget" and save them to a CSV file
    grep "Budget" output.log > "batch_size/batch_size_${batch_size}.csv"
done

# Clean up the output log file
rm output.log

