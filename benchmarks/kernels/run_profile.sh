#!/bin/bash

# Define the arrays for the parameters
batch_sizes=(1 2 4 8 16 32)
context_lengths=(512 1024 2048)
query_lengths=(4)
num_threads=(128 256 512)
num_partitions=(32 64 128 256 512)

# Print the CSV header
echo "Batch Size, Context Length, Query Length, Num Threads, Num Partitions, Original Time, Target Time, Speedup, Validation"

# Loop over all combinations of parameters
for batch_size in "${batch_sizes[@]}"; do
    for context_length in "${context_lengths[@]}"; do
        for query_length in "${query_lengths[@]}"; do
            for num_thread in "${num_threads[@]}"; do
                for num_partition in "${num_partitions[@]}"; do
                    # Run the script and parse the output
                    output=$(python3 bench_hj.py --batch-size $batch_size --context-len $context_length --query-len $query_length --num-threads $num_thread --partition-size $num_partition)
                    original_time=$(echo "$output" | grep "Original kernel running time:" | awk '{print $5}')
                    target_time=$(echo "$output" | grep "Target kernel running time:" | awk '{print $5}')
                    speedup=$(echo "$output" | grep "Speedup :" | awk '{print $3}')
                    validation=$(echo "$output" | grep "Validation success:" | awk '{print $3}')
                    
                    # Print the results in CSV format
                    echo "$batch_size, $context_length, $query_length, $num_thread, $num_partition, $original_time, $target_time, $speedup, $validation"
                done
            done
        done
    done
done