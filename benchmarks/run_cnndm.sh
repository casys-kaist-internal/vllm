#!/bin/bash

# Set the output file path
output_file="./results.csv"

# Set the number of prompts
num_prompts=1000

rm $output_file

echo -n "Base, " >> "$output_file"
python3 benchmark_sps_latency.py \
--num-prompts "$num_prompts" \
--engine base \
| grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"

# Loop through window sizes from 2 to 10
for window_size in {2..10}; do
    # Run the benchmark with sps engine and current window size
    echo -n "SpS $window_size, " >> "$output_file"
    python3 benchmark_sps_latency.py \
    --num-prompts "$num_prompts" \
    --engine sps \
    --draft-size "$window_size" \
    | grep "latency" | awk '{printf "%.3f\n", $3}' >> "$output_file"
done
