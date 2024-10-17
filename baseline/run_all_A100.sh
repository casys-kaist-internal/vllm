#!/bin/bash

set -e
set -o pipefail

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

# Get GPU name using Python script
gpu_name=$(python3 get_gpu_name.py)

# Cleanup the GPU name
gpu_name=$(echo "$gpu_name" | tr -d '[:space:]')

echo "GPU: $gpu_name"

# Model pairs to benchmark
declare -a models=(
    # "huggyllama/llama-7b,JackFram/llama-68m"
    "facebook/opt-6.7b,facebook/opt-125m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
)

# Common arguments
datasets=("finance")
temperatures=(0)
request_rates=(16)
draft_sizes_ar=(0)
draft_sizes_speculative=(1 3 5 7)
budget_seqs=(128)
prefill_schedule_mode="full_prefill"
colocate="False"
consolidated_attention="False"
drop_threshold="0"
budget_token="4096"

# Output CSV file
output_csv="baseline_pythia_sharegpt_$gpu_name.csv"

# Initialize CSV file
initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "Initializing CSV file: $output_csv"
        echo "gpu_name,target_model,draft_model,dataset,temperature,request_rate,draft_size,prefill_schedule_mode,budget_token,budget_seq,colocate,consolidated_attention,drop_threshold,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_token_latency,p99_token_latency,token_throughput,request_throughput,token_latency,preempt_flag" > "$output_csv"
    fi
}

# Initialize the CSV file
initialize_csv

# Function to check if configuration exists in CSV
configuration_exists_in_csv() {
    local gpu_name="$1"
    local target_model="$2"
    local draft_model="$3"
    local dataset="$4"
    local temperature="$5"
    local request_rate="$6"
    local draft_size="$7"
    local budget_seq="$8"  # Add budget_seq parameter

    # Skip header line (NR > 1)
    if awk -v OFS=',' -F', *' -v gpu_name="$gpu_name" \
            -v target_model="$target_model" \
            -v draft_model="$draft_model" \
            -v dataset="$dataset" \
            -v temperature="$temperature" \
            -v request_rate="$request_rate" \
            -v draft_size="$draft_size" \
            -v budget_seq="$budget_seq" '  # Include budget_seq in the check
        NR > 1 {
            for (i=1; i<=NF; i++) { gsub(/^ +| +$/, "", $i) }
            if ($1 == gpu_name && $2 == target_model && $3 == draft_model && $4 == dataset && $5 == temperature && $6 == request_rate && $7 == draft_size && $10 == budget_seq) {
                found = 1; exit
            }
        }
        END { exit !found }
    ' "$output_csv"; then
        return 0  # Configuration exists
    else
        return 1  # Configuration does not exist
    fi
}


# Function to extract values from the benchmark output
extract_values() {
    local log_file="$1"
    local result_line=$(grep 'Result' "$log_file")
    if [ -z "$result_line" ]; then
        echo "Error: No 'result' line found in output."
        return 1
    fi
    IFS=', ' read -ra metrics <<< "$result_line"
    request_throughput="${metrics[1]}"
    token_throughput="${metrics[2]}"
    token_latency="${metrics[3]}"
    p50_ttft="${metrics[4]}"
    p99_ttft="${metrics[5]}"
    p50_tpot="${metrics[6]}"
    p99_tpot="${metrics[7]}"
    p50_token_latency="${metrics[8]}"
    p99_token_latency="${metrics[9]}"
    preempt_flag="${metrics[10]}"
}

# AutoRegressive Decoding
for budget_seq in ${budget_seqs[@]}; do
    for model_pair in "${models[@]}"; do
        IFS=',' read -r target_model draft_model <<< "$model_pair"
        for dataset in "${datasets[@]}"; do    
            for request_rate in "${request_rates[@]}"; do
                for draft_size in "${draft_sizes_ar[@]}"; do
                    temperature="0"  # AR decoding always uses temperature 0
                    log_file="logs/AR_${target_model}_${draft_model}_${dataset}_${request_rate}_${draft_size}_${budget_seq}.log"
                    mkdir -p $(dirname "$log_file")

                    # Check if configuration already exists
                    if configuration_exists_in_csv "$gpu_name" "$target_model" "$draft_model" "$dataset" "$temperature" "$request_rate" "$draft_size" "$budget_seq"; then
                        echo "Configuration already exists in CSV. Skipping."
                        continue
                    fi

                    # Run the benchmark
                    python3 baseline_ar.py \
                        --dataset "$dataset" \
                        --target-model "$target_model" \
                        --draft-model "$draft_model" \
                        --draft-size "$draft_size" \
                        --budget-token "$budget_token" \
                        --budget-seq "$budget_seq" \
                        --request-rate "$request_rate" > "$log_file" 2>&1

                    # Extract values from the log file
                    if ! extract_values "$log_file"; then
                        echo "Failed to extract values. Skipping."
                        continue
                    fi

                    # Append results to CSV
                    echo "$gpu_name,$target_model,$draft_model,$dataset,$temperature,$request_rate,$draft_size,$prefill_schedule_mode,$budget_token,$budget_seq,$colocate,$consolidated_attention,$drop_threshold,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_token_latency,$p99_token_latency,$token_throughput,$request_throughput,$token_latency,$preempt_flag" >> "$output_csv"
                done
            done
        done
    done
done

# Speculative Decoding
# for budget_seq in ${budget_seqs[@]}; do
#     for model_pair in "${models[@]}"; do
#         IFS=',' read -r target_model draft_model <<< "$model_pair"
#         for dataset in "${datasets[@]}"; do
#             for temperature in "${temperatures[@]}"; do
#                 for request_rate in "${request_rates[@]}"; do
#                     for draft_size in "${draft_sizes_speculative[@]}"; do
#                         log_file="logs/speculative_${target_model}_${draft_model}_${dataset}_${temperature}_${request_rate}_${draft_size}_${budget_seq}.log"
#                         mkdir -p $(dirname "$log_file")

#                         # Check if configuration already exists
#                         if configuration_exists_in_csv "$gpu_name" "$target_model" "$draft_model" "$dataset" "$temperature" "$request_rate" "$draft_size" "$budget_seq"; then
#                             echo "Configuration already exists in CSV. Skipping."
#                             continue
#                         fi

#                         # Run the benchmark
#                         python3 baseline.py \
#                             --dataset "$dataset" \
#                             --target-model "$target_model" \
#                             --draft-model "$draft_model" \
#                             --draft-size "$draft_size" \
#                             --budget-seq "$budget_seq" \
#                             --budget-token "$budget_token" \
#                             --temperature "$temperature" \
#                             --request-rate "$request_rate" > "$log_file" 2>&1

#                         # Extract values from the log file
#                         if ! extract_values "$log_file"; then
#                             echo "Failed to extract values. Skipping."
#                             continue
#                         fi

#                         # Append results to CSV
#                         echo "$gpu_name,$target_model,$draft_model,$dataset,$temperature,$request_rate,$draft_size,$prefill_schedule_mode,$budget_token,$budget_seq,$colocate,$consolidated_attention,$drop_threshold,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_token_latency,$p99_token_latency,$token_throughput,$request_throughput,$token_latency,$preempt_flag" >> "$output_csv"
#                     done
#                 done
#             done
#         done
#     done
# done

# echo "All benchmarks completed."
