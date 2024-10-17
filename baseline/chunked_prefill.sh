#!/bin/bash

# Get GPU name using Python script
gpu_name=$(python3 get_gpu_name.py)

# Cleanup the GPU name
gpu_name=$(echo $gpu_name | tr -d '[:space:]')

echo "GPU: $gpu_name"

# Model pairs to benchmark
declare -a models=(
    # Uncomment the models you want to benchmark
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-125m"
    "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # Add more model pairs as needed
)

# Common arguments
# Define the request rates, draft sizes, and temperatures
#request_rates=(2 4 6 8 10 12 14 16 18 20)
request_rates=(6)
draft_sizes_ar=(0)
draft_sizes_speculative=(1 3 5 7)
temperatures=(0 0.25 0.5 0.75 -1)

# Define other default arguments (adjust as necessary)
datasets=("sharegpt")

# Output CSV file
output_csv="chunked_prefill_A6000.csv"

# Initialize the CSV file with the header if it doesn't exist
if [ ! -f "$output_csv" ]; then
    echo "Result,GPU Name,Target Model,Draft Model,Dataset,Temperature,Request Rate,Draft Size,Request Throughput (reqs/s),Token Throughput (tokens/s),Token Latency (s/token),P50 TTFT (s),P99 TTFT (s),P50 TPOT (s/token),P99 TPOT (s/token),P50 Token Latency (s/token),P99 Token Latency (s/token),Preempt Flag" > "$output_csv"
fi

# Calculate total number of iterations for progress tracking
total_iterations=$(( ${#models[@]} * ${#datasets[@]} * ${#request_rates[@]} * ( ${#draft_sizes_ar[@]} ) ))
iteration=0

# AutoRegressive Decoding
# Loop through each model pair, dataset, request rate, and draft size combination
for model_pair in "${models[@]}"; do
  IFS=',' read -r target_model draft_model <<< "$model_pair"
  for dataset in "${datasets[@]}"; do
    for request_rate in "${request_rates[@]}"; do
      for draft_size in "${draft_sizes_ar[@]}"; do
        iteration=$((iteration + 1))
        ./slack "Progress: $iteration/$total_iterations"
        echo "Running AR benchmark with target model: $target_model, draft model: $draft_model, dataset: $dataset, request rate: $request_rate, draft size: $draft_size"
        
        # Run the benchmark script and append the output to the CSV file
        python3 baseline_ar_chunked_prefill.py \
            --dataset "$dataset" \
            --target-model "$target_model" \
            --draft-model "$draft_model" \
            --draft-size "$draft_size" \
            --request-rate "$request_rate" | grep "Result" >> "$output_csv"
        
        echo "Saved results for target model: $target_model, draft model: $draft_model, dataset: $dataset, request rate: $request_rate, draft size: $draft_size"
      done
    done
  done
done

echo "All AutoRegressive benchmarks completed."

# # Speculative Decoding
# # Loop through each model pair, dataset, temperature, request rate, and draft size combination
# for model_pair in "${models[@]}"; do
#   IFS=',' read -r target_model draft_model <<< "$model_pair"
#   for dataset in "${datasets[@]}"; do
#     for temperature in "${temperatures[@]}"; do
#       for request_rate in "${request_rates[@]}"; do
#         for draft_size in "${draft_sizes_speculative[@]}"; do
#           iteration=$((iteration + 1))
#           ./slack "Progress: $iteration/$total_iterations"
#           echo "Running Speculative benchmark with target model: $target_model, draft model: $draft_model, dataset: $dataset, temperature: $temperature, request rate: $request_rate, draft size: $draft_size"
          
#           # Run the benchmark script and append the output to the CSV file
#           python3 baseline.py \
#               --dataset "$dataset" \
#               --target-model "$target_model" \
#               --draft-model "$draft_model" \
#               --draft-size "$draft_size" \
#               --temperature "$temperature" \
#               --request-rate "$request_rate" | grep "Result" >> "$output_csv"
          
#           echo "Saved results for target model: $target_model, draft model: $draft_model, dataset: $dataset, temperature: $temperature, request rate: $request_rate, draft size: $draft_size"
#         done
#       done
#     done
#   done
# done

# echo "All Speculative Decoding benchmarks completed."

# # Disable Speculative Decoding by Batch Size
# # Define disable size array
# disable_size=(32 64 128)

# # Loop through each model pair, request rate, draft size, and disable size combination
# for model_pair in "${models[@]}"; do
#   IFS=',' read -r target_model draft_model <<< "$model_pair"
#   for request_rate in "${request_rates[@]}"; do
#     for draft_size in "${draft_sizes_speculative[@]}"; do
#       for disable in "${disable_size[@]}"; do
#         iteration=$((iteration + 1))
#         ./slack "Progress: $iteration/$total_iterations"
#         echo "Running benchmark with target model: $target_model, draft model: $draft_model, request rate: $request_rate, draft size: $draft_size, disable size: $disable"
        
#         # Run the benchmark script and append the output to the CSV file
#         python3 baseline_specdis.py \
#             --dataset "$dataset" \
#             --target-model "$target_model" \
#             --draft-model "$draft_model" \
#             --draft-size "$draft_size" \
#             --temperature "$temperature" \
#             --speculative-disable-by-batch-size "$disable" \
#             --request-rate "$request_rate" | grep "Result" >> "$output_csv"
        
#         echo "Saved results for target model: $target_model, draft model: $draft_model, request rate: $request_rate, draft size: $draft_size, disable size: $disable"
#       done
#     done
#   done
# done

# echo "All benchmarks completed."
