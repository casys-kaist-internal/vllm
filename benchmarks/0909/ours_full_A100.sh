#!/bin/bash

set -e
set -o pipefail

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

# Model pairs to benchmark
declare -a models=(
    # Uncomment the models you want to benchmark
    # "facebook/opt-13b,facebook/opt-125m"
    "facebook/opt-6.7b,facebook/opt-125m"
    # "huggyllama/llama-7b,JackFram/llama-68m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # Add more model pairs as needed
)

# Configurations
datasets=("sharegpt")
temperatures=(0.75)
# request_rates=(1 2 4 8 16 32)
# request_rates=(4 8 16 24 32)
request_rates=(10)
draft_sizes=(7)
prefill_schedule_modes=("full_prefill")
budget_tokens=(4096)
budget_seqs=(128)
colocates=(true)
consolidated_attentions=(true)
drop_thresholds=(0.3)

# Paths
python_script="benchmark_serving.py"
output_csv="figures/ours_A100_pythia_sharegpt_10_16_.csv"

# Create directory if it doesn't exist
mkdir -p figures
mkdir -p logs

# Function to get GPU name
get_gpu_name() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        # Replace spaces with underscores for file naming
        gpu_name=${gpu_name// /_}
    else
        gpu_name="Unknown"
    fi
}

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "Initializing CSV file: $output_csv"
        echo "gpu_name,target_model,draft_model,dataset,temperature,request_rate,draft_size,prefill_schedule_mode,budget_token,budget_seq,colocate,consolidated_attention,drop_threshold,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_token_latency,p99_token_latency,token_throughput,request_throughput,token_latency,preempt_flag" > "$output_csv"
    else
        echo "Warning: $output_csv already exists."
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

run_benchmark() {
    local dataset="$1"
    local temperature="$2"
    local request_rate="$3"
    local draft_size="$4"
    local prefill_schedule_mode="$5"
    local budget_token="$6"
    local budget_seq="$7"
    local colocate="$8"
    local consolidated_attention="$9"
    local drop_threshold="${10}"
    local target_model="${11}"
    local draft_model="${12}"

    # Adjust parameters based on conditions
    if [ "$draft_size" = "0" ]; then
        consolidated_attention="false"
        colocate="false"
        # Skip if drop_threshold > 0
        if [ "$drop_threshold" != "0" ]; then
            return
        fi
    fi

    # Check if configuration already exists in CSV
    if awk -F, -v gpu_name="$gpu_name" \
            -v target_model="$target_model" \
            -v draft_model="$draft_model" \
            -v dataset="$dataset" \
            -v temperature="$temperature" \
            -v request_rate="$request_rate" \
            -v draft_size="$draft_size" \
            -v prefill_schedule_mode="$prefill_schedule_mode" \
            -v budget_token="$budget_token" \
            -v budget_seq="$budget_seq" \
            -v colocate="$colocate" \
            -v consolidated_attention="$consolidated_attention" \
            -v drop_threshold="$drop_threshold" '
        NR > 1 &&
        $1 == gpu_name &&
        $2 == target_model &&
        $3 == draft_model &&
        $4 == dataset &&
        $5 == temperature &&
        $6 == request_rate &&
        $7 == draft_size &&
        $8 == prefill_schedule_mode &&
        $9 == budget_token &&
        $10 == budget_seq &&
        $11 == colocate &&
        $12 == consolidated_attention &&
        $13 == drop_threshold {
            found = 1; exit
        }
        END { exit !found }
    ' "$output_csv"; then
        echo "Configuration already exists in CSV. Skipping."
        return
    fi

    echo "Running benchmark:"
    echo "Dataset: $dataset, Temperature: $temperature, Request Rate: $request_rate"
    echo "Draft Size: $draft_size, Prefill Schedule Mode: $prefill_schedule_mode"
    echo "Budget Token: $budget_token, Budget Seq: $budget_seq"
    echo "Colocate: $colocate, Consolidated Attention: $consolidated_attention"
    echo "Drop Threshold: $drop_threshold"
    echo "Target Model: $target_model, Draft Model: $draft_model"
    echo "GPU Name: $gpu_name"

    # Build command line arguments
    args=(
        "$python_script"
        "--dataset" "$dataset"
        "--temperature" "$temperature"
        "--request-rate" "$request_rate"
        "--draft-size" "$draft_size"
        "--prefill-schedule-mode" "$prefill_schedule_mode"
        "--budget-token" "$budget_token"
        "--budget-seq" "$budget_seq"
        "--target-model" "$target_model"
        "--draft-model" "$draft_model"
    )

    [ "$colocate" = "true" ] && args+=("--colocate")
    [ "$consolidated_attention" = "true" ] && args+=("--consolidated-attention")
    [ "$drop_threshold" != "0" ] && args+=("--selective-validation" "--drop-threshold" "$drop_threshold")

    # Save the last run command to a file for reference
    echo "python ${args[*]}" > last_run.sh
    echo "Command: python ${args[*]}"

    # Create a safe log file name based on current configuration
    local target_model_name="${target_model##*/}"
    local draft_model_name="${draft_model##*/}"

    # Replace any non-alphanumeric characters with underscores for safety
    target_model_name=${target_model_name//[^a-zA-Z0-9]/_}
    draft_model_name=${draft_model_name//[^a-zA-Z0-9]/_}

    timestamp=$(date +"%Y%m%d_%H%M%S")

    log_file="logs/benchmark_${timestamp}_${gpu_name}_${dataset}_temp${temperature}_rate${request_rate}_draft${draft_size}_budgettoken${budget_token}_budgetseq${budget_seq}_colocate${colocate}_attn${consolidated_attention}_drop${drop_threshold}_${target_model_name}_${draft_model_name}.log"

    # Shorten log file name if it becomes too long (optional)
    if [ ${#log_file} -gt 255 ]; then
        log_file_hash=$(echo -n "$log_file" | md5sum | awk '{print $1}')
        log_file="logs/benchmark_${log_file_hash}.log"
    fi

    if ! python "${args[@]}" > "$log_file" 2>&1; then
        echo "Error: Benchmark script failed. See $log_file for details."
        return 1
    fi

    # Extract values from log file
    if ! extract_values "$log_file"; then
        echo "Error: Failed to extract values from output. See $log_file for details."
        return 1
    fi

    # Append results to CSV
    echo "$gpu_name,$target_model,$draft_model,$dataset,$temperature,$request_rate,$draft_size,$prefill_schedule_mode,$budget_token,$budget_seq,$colocate,$consolidated_attention,$drop_threshold,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_token_latency,$p99_token_latency,$token_throughput,$request_throughput,$token_latency,$preempt_flag" >> "$output_csv"
}

# Initialize CSV file
initialize_csv

# Get GPU name
get_gpu_name

# Calculate total runs for progress tracking
total_runs=$(( ${#datasets[@]} * ${#temperatures[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#prefill_schedule_modes[@]} * ${#budget_tokens[@]} * ${#budget_seqs[@]} * ${#colocates[@]} * ${#consolidated_attentions[@]} * ${#drop_thresholds[@]} * ${#models[@]} ))
current_run=1

# Main loop over all parameter combinations
for model_pair in "${models[@]}"; do
    IFS=',' read -r target_model draft_model <<< "$model_pair"
    for dataset in "${datasets[@]}"; do
        for temperature in "${temperatures[@]}"; do
            for request_rate in "${request_rates[@]}"; do
                for draft_size in "${draft_sizes[@]}"; do
                    for prefill_schedule_mode in "${prefill_schedule_modes[@]}"; do
                        for budget_token in "${budget_tokens[@]}"; do
                            for budget_seq in "${budget_seqs[@]}"; do
                                for colocate in "${colocates[@]}"; do
                                    for consolidated_attention in "${consolidated_attentions[@]}"; do
                                        for drop_threshold in "${drop_thresholds[@]}"; do
                                            echo "Progress: [$current_run/$total_runs]"
                                            if run_benchmark "$dataset" "$temperature" "$request_rate" "$draft_size" "$prefill_schedule_mode" "$budget_token" "$budget_seq" "$colocate" "$consolidated_attention" "$drop_threshold" "$target_model" "$draft_model"; then
                                                echo "Benchmark completed successfully."
                                                ./slack "Progress: [$current_run/$total_runs] Success"
                                            else
                                                echo "Benchmark failed."
                                                ./slack "Progress: [$current_run/$total_runs] Fail"
                                            fi
                                            ((current_run++))
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
