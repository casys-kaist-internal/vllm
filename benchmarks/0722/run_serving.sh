#!/bin/bash

# Colocate True False 
# Target Attention True 

# Emulate accept probs 0.7
# input_len + output_len = 2048
# 1) input_len : output_len = 0.25 : 0.75
# 2) input_len : output_len = 0.5 : 0.5
# 3) input_len : output_len = 0.75 : 0.25

# Decode only (Prioritize prefill)
# Budget 2048

# Decode and chunked prefill (Prioritize decode)
# Decode and full prefill (Prioritize decode + prefill)
# Budget 512, 1024, 2048

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

# Configurations
request_rates=(8 16 24 28)
draft_sizes=(0 2 4 6)
chunk_prefills=(false)
colocates=(false)
target_attentions=(false)
datasets=("finance")
temperatures=(0)
drop_thresholds=(0)
budgets=(2048)

# Paths
python_script="benchmark_serving.py"
output_csv="benchmark_results_finance_0720.csv"

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "dataset,request_rate,draft_size,chunk_prefill,colocate,target_attention,budget,temperature,drop_threshold,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_tpt,p99_tpt,throughput,latency,preempt_flag" > "$output_csv"
    else
        echo "Error: $output_csv already exists."
        exit 1
    fi
}


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
    preempt_flag=$(echo "$output" | awk -F', ' '/result/{print $10}')
}

# Run the benchmark for the current configuration
run_benchmark() {
    local dataset="$1"
    local request_rate="$2"
    local draft_size="$3"
    local chunk_prefill="$4"
    local colocate="$5"
    local target_attention="$6"
    local budget="$7"
    local temperature="$8"
    local drop_threshold="$9"

    if [ "$draft_size" = "0" ]; then
        # colocate = true and target_attention = true only when draft_size > 0
        colocate="false"
        target_attention="false"
        # if [ "$colocate" = "true" ] || [ "$target_attention" = "true" ]; then
        #     return
        # fi
        # skip if drop_threshold > 0
        if [ "$drop_threshold" != "0" ]; then
            return
        fi
    fi


    echo "Running benchmark for $dataset,  request_rate: $request_rate, draft_size: $draft_size, chunk_prefill: $chunk_prefill, colocate: $colocate, target_attention: $target_attention, budget: $budget, temperature: $temperature, drop_threshold: $drop_threshold"


    chunk_prefill_flag=""
    [ "$chunk_prefill" = "true" ] && chunk_prefill_flag="--chunked-prefill"
    colocate_flag=""
    [ "$colocate" = "true" ] && colocate_flag="--colocate"
    target_attention_flag=""
    [ "$target_attention" = "true" ] && target_attention_flag="--target-attention"

    # save the last running python line in a file
    echo "python "$python_script" --dataset "$dataset" --request-rate "$request_rate" --draft-size "$draft_size" $chunk_prefill_flag $colocate_flag $target_attention_flag --budget $budget --temperature $temperature --drop-threshold $drop_threshold " > last_run.sh

    local output=$(python "$python_script" --dataset "$dataset" --request-rate "$request_rate" --draft-size "$draft_size" $chunk_prefill_flag $colocate_flag $target_attention_flag  --budget $budget --temperature $temperature --drop-threshold $drop_threshold --enforce-eager)
    
    extract_values "$output"
    echo "$dataset,$request_rate,$draft_size,$chunk_prefill,$colocate,$target_attention,$budget,$temperature,$drop_threshold,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency,$preempt_flag" >> "$output_csv"
    ./slack "dataset: $dataset, request_rate: $request_rate, draft_size: $draft_size, chunk_prefill: $chunk_prefill, colocate: $colocate, attention: $target_attention, budget: $budget, temperature: $temperature, drop_threshold: $drop_threshold, throughput: $throughput, latency: $latency, preempted: $preempt_flag"
}

# Initialize
initialize_csv

total_runs=$(( ${#datasets[@]} * ${#temperatures[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#chunk_prefills[@]} * ${#colocates[@]} * ${#target_attentions[@]} * ${#budgets[@]} * ${#drop_thresholds[@]}))
current_run=0

# Main Loop
for dataset in "${datasets[@]}"; do
    for temperature in "${temperatures[@]}"; do
        for budget in "${budgets[@]}"; do
            for draft_size in "${draft_sizes[@]}"; do
                for chunk_prefill in "${chunk_prefills[@]}"; do
                    for request_rate in "${request_rates[@]}"; do
                        for colocate in "${colocates[@]}"; do
                            for target_attention in "${target_attentions[@]}"; do
                                for drop_threshold in "${drop_thresholds[@]}"; do
                                    echo "[${current_run}/${total_runs}]"
                                    ./slack "[${current_run}/${total_runs}]"
                                    run_benchmark "$dataset" "$request_rate" "$draft_size" "$chunk_prefill" "$colocate" "$target_attention" "$budget" "$temperature" "$drop_threshold"
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
