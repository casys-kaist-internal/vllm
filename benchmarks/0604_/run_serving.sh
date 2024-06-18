#!/bin/bash

# Define the different configurations
# request_rates=(4 8 12 16 20 24 28 32)
request_rates=(1 2 4 8 16 32 64)
# draft_sizes=(0 1 2 3 4 5 6 7)
draft_sizes=(0 4)
chunk_prefills=(false true)
collocates=(false true)
datasets=("finance")

# Path to the Python script
python_script="benchmark_serving.py"

# Output CSV file
output_csv="benchmark_results.csv"

# Write the header to the CSV file
echo "dataset,request_rate,draft_size,chunk_prefill,collocate,total_time,throughput,avg_latency,avg_per_token_latency,avg_per_output_token_latency" > $output_csv

# Function to extract values from the benchmark output
extract_values() {
    local output=$1
    total_time=$(echo "$output" | grep "Total time" | awk '{print $3}')
    throughput=$(echo "$output" | grep "Throughput" | awk '{print $2}')
    avg_latency=$(echo "$output" | grep "Average latency:" | awk '{print $3}')
    avg_per_token_latency=$(echo "$output" | grep "Average latency per token:" | awk '{print $5}')
    avg_per_output_token_latency=$(echo "$output" | grep "Average latency per output token:" | awk '{print $6}')
}

# Run the benchmark for each combination of parameters
for dataset in "${datasets[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        for draft_size in "${draft_sizes[@]}"; do
            for chunk_prefill in "${chunk_prefills[@]}"; do
                chunk_prefill_flag=""
                if [ "$chunk_prefill" = true ]; then
                    chunk_prefill_flag="--chunked-prefill"
                fi
                
                if [ "$draft_size" -eq 0 ]; then
                    # Collocate should be disabled when draft size is 0
                    collocate=false
                    echo "Running with dataset=$dataset, request_rate=$request_rate, draft_size=$draft_size, chunk_prefill=$chunk_prefill, collocate=$collocate"
                    output=$(python $python_script --dataset $dataset --request-rate $request_rate --draft-size $draft_size $chunk_prefill_flag)
                    extract_values "$output"
                    echo "$dataset,$request_rate,$draft_size,$chunk_prefill,$collocate,$total_time,$throughput,$avg_latency,$avg_per_token_latency,$avg_per_output_token_latency" >> $output_csv
                else
                    for collocate in "${collocates[@]}"; do
                        collocate_flag=""
                        if [ "$collocate" = true ]; then
                            collocate_flag="--collocate"
                        fi
                        echo "Running with dataset=$dataset, request_rate=$request_rate, draft_size=$draft_size, chunk_prefill=$chunk_prefill, collocate=$collocate"
                        output=$(python $python_script --dataset $dataset --request-rate $request_rate --draft-size $draft_size $chunk_prefill_flag $collocate_flag)
                        extract_values "$output"
                        echo "$dataset,$request_rate,$draft_size,$chunk_prefill,$collocate,$total_time,$throughput,$avg_latency,$avg_per_token_latency,$avg_per_output_token_latency" >> $output_csv
                    done
                fi
            done
        done
    done
done
