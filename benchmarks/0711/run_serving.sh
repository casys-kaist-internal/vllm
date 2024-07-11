#!/bin/bash

# Configurations
# request_rates=(16 18 20)
request_rates=(8 12 16 20)
draft_sizes=(0 4 7)
chunk_prefills=(false true)
colocates=(false true)
target_attentions=(true)
datasets=("finance")
output_lens=(0)  # 0 means use original output length
temperatures=(0)

# Paths
python_script="benchmark_serving.py"
output_csv="benchmark_results.csv"

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "dataset,temperature,output_len,request_rate,draft_size,chunk_prefill,colocate,target_attention,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_tpt,p99_tpt,throughput,latency,preempt_flag" > "$output_csv"
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
    local output_len="$2"
    local request_rate="$3"
    local draft_size="$4"
    local budget="$5"
    local chunk_prefill_flag="$6"
    local colocate_flag="$7"
    local target_attention_flag="$8"
    local output_len_flag="$9"

    local output=$(python "$python_script" --dataset "$dataset" --request-rate "$request_rate" --draft-size "$draft_size" --budget "$budget" $chunk_prefill_flag $colocate_flag $target_attention_flag $output_len_flag --temperature "$temperature")
    extract_values "$output"
    echo "$dataset,$temperature,$output_len,$request_rate,$draft_size,$budget,$chunk_prefill,$colocate,$target_attention,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency,$preempt_flag" >> "$output_csv"
    ./slack "dataset: $dataset, temperature: $temperature, output_len: $output_len, request_rate: $request_rate, draft_size: $draft_size, budget: $budget, chunk_prefill: $chunk_prefill, colocate: $colocate, attention: $target_attention, throughput: $throughput, latency: $latency, preempted: $preempt_flag"
}

# Initialize
initialize_csv

# Main Loop
for dataset in "${datasets[@]}"; do
    for temperature in "${temperatures[@]}"; do
        for output_len in "${output_lens[@]}"; do
            for draft_size in "${draft_sizes[@]}"; do
                for chunk_prefill in "${chunk_prefills[@]}"; do
                    if [ "$chunk_prefill" = "true" ]; then
                        budgets=(256 512 1024 2048 4096)
                    else
                        budgets=(2048 4096)
                    fi
                    for budget in "${budgets[@]}"; do
                        for request_rate in "${request_rates[@]}"; do
                            output_len_flag=""
                            [ "$output_len" -ne 0 ] && output_len_flag="--output-len $output_len"
                            chunk_prefill_flag=""
                            [ "$chunk_prefill" = "true" ] && chunk_prefill_flag="--chunked-prefill"
                            if [ "$draft_size" -eq 0 ]; then
                                colocate=false
                                target_attention=false
                                run_benchmark "$dataset" "$output_len" "$request_rate" "$draft_size" "$budget" "$chunk_prefill_flag" "" "" "$output_len_flag" "$temperature"
                            else
                                for colocate in "${colocates[@]}"; do
                                    for target_attention in "${target_attentions[@]}"; do
                                        colocate_flag=""
                                        [ "$colocate" = "true" ] && colocate_flag="--colocate"
                                        target_attention_flag=""
                                        [ "$target_attention" = "true" ] && target_attention_flag="--target-attention"
                                        run_benchmark "$dataset" "$output_len" "$request_rate" "$draft_size" "$budget" "$chunk_prefill_flag" "$colocate_flag" "$target_attention_flag" "$output_len_flag" "$temperature"
                                    done
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done

# Colocate True False 
# Target Attention True 

# Emulate accept probs 0.7
# input_len + output_len = 2048
# 1) input_len : output_len = 0.25 : 0.75
# 2) input_len : output_len = 0.5 : 0.5
# 3) input_len : output_len = 0.75 : 0.25

# Decode only (Prioritize prefill)
# Budget 16384

# Decode and chunked prefill (Prioritize decode)
# Decode and full prefill (Prioritize decode + prefill)
# Budget 256, 512, 1024, 2048, 4096, 8192, 16384