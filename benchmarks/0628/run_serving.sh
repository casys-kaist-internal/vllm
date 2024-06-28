#!/bin/bash

# Configurations
# request_rates=(16 18 20)
request_rates=(4 8 12 16)
draft_sizes=(0)
chunk_prefills=(false true)
colocates=(false)
target_attentions=(false)
datasets=("finance" "apps" "sharegpt")
output_lens=(0)  # 0 means use original output length

# Paths
python_script="benchmark_serving.py"
output_csv="benchmark_results_compare_with_base.csv"

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "dataset,output_len,request_rate,draft_size,chunk_prefill,colocate,target_attention,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_tpt,p99_tpt,throughput,latency,preempt_flag" > "$output_csv"
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
    local chunk_prefill_flag="$5"
    local colocate_flag="$6"
    local target_attention_flag="$7"
    local output_len_flag="$8"

    local output=$(python "$python_script" --dataset "$dataset" --request-rate "$request_rate" --draft-size "$draft_size" $chunk_prefill_flag $colocate_flag $target_attention_flag $output_len_flag)
    extract_values "$output"
    echo "$dataset,$output_len,$request_rate,$draft_size,$chunk_prefill,$colocate,$target_attention,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency,$preempt_flag" >> "$output_csv"
    ./slack "$dataset,$output_len,$request_rate,$draft_size,$chunk_prefill,$colocate,$target_attention, $throughput, $latency, $preempt_flag"
}

# Initialize
initialize_csv

total_runs=$(( ${#datasets[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#chunk_prefills[@]} * ${#colocates[@]} * ${#target_attentions[@]} * ${#output_lens[@]} ))
current_run=0

# Main Loop
for dataset in "${datasets[@]}"; do
    for output_len in "${output_lens[@]}"; do
        for draft_size in "${draft_sizes[@]}"; do
            for chunk_prefill in "${chunk_prefills[@]}"; do
                for request_rate in "${request_rates[@]}"; do
                    output_len_flag=""
                    [ "$output_len" -ne 0 ] && output_len_flag="--output-len $output_len"
                    chunk_prefill_flag=""
                    [ "$chunk_prefill" = "true" ] && chunk_prefill_flag="--chunked-prefill"
                    if [ "$draft_size" -eq 0 ]; then
                        colocate=false
                        target_attention=false
                        run_benchmark "$dataset" "$output_len" "$request_rate" "$draft_size" "$chunk_prefill_flag" "" "" "$output_len_flag"
                        ((current_run++))
                        ./slack "[${current_run}/${total_runs}]"
                    else
                        for colocate in "${colocates[@]}"; do
                            for target_attention in "${target_attentions[@]}"; do
                                colocate_flag=""
                                [ "$colocate" = "true" ] && colocate_flag="--colocate"
                                target_attention_flag=""
                                [ "$target_attention" = "true" ] && target_attention_flag="--target-attention"
                                run_benchmark "$dataset" "$output_len" "$request_rate" "$draft_size" "$chunk_prefill_flag" "$colocate_flag" "$target_attention_flag" "$output_len_flag"
                                ((current_run++))
                                ./slack "[${current_run}/${total_runs}]"
                            done
                        done
                    fi
                done
            done
        done
    done
done
