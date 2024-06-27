#!/bin/bash

# Define the different configurations
# request_rates=(4 8 12 16 20 24 28 32)
request_rates=(1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8)
# draft_sizes=(0 1 2 3 4 5 6 7)
draft_sizes=(0 4 7)
chunk_prefills=(false true)
colocates=(false true)
datasets=("finance")

# Path to the Python script
python_script="benchmark_serving.py"

# Output CSV file
output_csv="benchmark_results.csv"

# Write the header to the CSV file
echo "dataset,request_rate,draft_size,chunk_prefill,colocate,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_tpt,p99_tpt,throughput,latency" > $output_csv

# Function to extract values from the benchmark output
extract_values() {
    local output="$1"
    p50_ttft=$(echo "$output" | awk -F', ' '/result/{print $2}')
    p99_ttft=$(echo "$output" | awk -F', ' '/result/{print $3}')
    p50_tpot=$(echo "$output" | awk -F', ' '/result/{print $4}')
    p99_tpot=$(echo "$output" | awk -F', ' '/result/{print $5}')
    p50_tpt=$(echo "$output" | awk -F', ' '/result/{print $6}')
    p99_tpt=$(echo "$output" | awk -F', ' '/result/{print $7}')
    throughput=$(echo "$output" | awk -F', ' '/result/{print $8}')
    latency=$(echo "$output" | awk -F', ' '/result/{print $9}')
}

total_runs=$(( ${#datasets[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#chunk_prefills[@]} * ${#colocates[@]} ))
current_run=0
# Run the benchmark for each combination of parameters
for dataset in "${datasets[@]}"; do
    for request_rate in "${request_rates[@]}"; do
        for draft_size in "${draft_sizes[@]}"; do
            for chunk_prefill in "${chunk_prefills[@]}"; do
                chunk_prefill_flag=""
                if [ "$chunk_prefill" = "true" ]; then
                    chunk_prefill_flag="--chunked-prefill"
                fi
                
                if [ "$draft_size" -eq 0 ]; then
                    colocate=false
                    output=$(python $python_script --dataset $dataset --request-rate $request_rate --draft-size $draft_size $chunk_prefill_flag)
                    extract_values "$output"
                    echo "$dataset,$request_rate,$draft_size,$chunk_prefill,$colocate,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency" >> $output_csv
                    ./slack "$dataset,$request_rate,$draft_size,$chunk_prefill,$colocate, $throughput, $latency"
                    ((current_run++))
                    ./slack "[${current_run}/${total_runs}]"
                else
                    for colocate in "${colocates[@]}"; do
                        colocate_flag=""
                        if [ "$colocate" = "true" ]; then
                            colocate_flag="--colocate"
                        fi
                        output=$(python $python_script --dataset $dataset --request-rate $request_rate --draft-size $draft_size $chunk_prefill_flag $colocate_flag)
                        extract_values "$output"
                        echo "$dataset,$request_rate,$draft_size,$chunk_prefill,$colocate,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency" >> $output_csv
                        ./slack "$dataset,$request_rate,$draft_size,$chunk_prefill,$colocate, $throughput, $latency"
                        ((current_run++))
                        ./slack "[${current_run}/${total_runs}]"
                    done
                fi
            done
        done
    done
done

