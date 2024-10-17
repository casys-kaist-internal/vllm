#!/bin/bash

# Define the request rates and draft sizes
request_rates=(1 2 4 8 16 24 32)
draft_sizes=(0)

# Define other default arguments (adjust as necessary)
datasets=("sharegpt" "finance" "gsm8k")
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"

# Output CSV file
output_csv="benchmark_results_ar.csv"

# Initialize the CSV file with the header if it doesn't exist
if [ ! -f "$output_csv" ]; then
    echo "GPU Name,Target Model,Draft Model,Dataset,Temperature,Request Rate,Draft Size,Request Throughput (reqs/s),Token Throughput (tokens/s),Token Latency (s/token),P50 TTFT (s),P99 TTFT (s),P50 TPOT (s/token),P99 TPOT (s/token),P50 Token Latency (s/token),P99 Token Latency (s/token),Preempt Flag" > "$output_csv"
fi

# Loop through each request rate and draft size combination
for request_rate in "${request_rates[@]}"; do
  for draft_size in "${draft_sizes[@]}"; do
    echo "Running benchmark with request rate: $request_rate and draft size: $draft_size"
    
    # Run the benchmark script and append the output to the CSV file
    python3 baseline_ar.py \
        --dataset $dataset \
        --target-model $target_model \
        --draft-model $draft_model \
        --draft-size $draft_size \
        --request-rate $request_rate | tail -n 1 >> "$output_csv"
    
    echo "Saved results for request rate: $request_rate and draft size: $draft_size"
  done
done

echo "All benchmarks completed."
