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

# Configurations
request_rates=(8 16 24 28)
draft_sizes=(0 4 7)
chunk_prefills=(false true)
colocates=(false true)
target_attentions=(true)
datasets=("dummy")
temperatures=(0)

input_output_lens_pairs=(
    "128 256"
    "256 256"
    "384 256"
)
budgets=(512 1024 2048)
demote_spec_tokens=(false true)

# Paths
python_script="benchmark_serving.py"
output_csv="benchmark_results.csv"

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "dataset,input_len,output_len,request_rate,draft_size,chunk_prefill,colocate,target_attention,budget,demote,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_tpt,p99_tpt,throughput,latency,preempt_flag" > "$output_csv"
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
    local input_len="$2"
    local output_len="$3"
    local request_rate="$4"
    local draft_size="$5"
    local chunk_prefill="$6"
    local colocate="$7"
    local target_attention="$8"
    local budget="$9"
    local demote="${10}"

    echo "Running benchmark for $dataset, input_len: $input_len, output_len: $output_len, request_rate: $request_rate, draft_size: $draft_size, chunk_prefill: $chunk_prefill, colocate: $colocate, target_attention: $target_attention, budget: $budget, demote: $demote"

    if [ "$draft_size" = "0" ]; then
        if [ "$colocate" = "true" ]; then
            echo "Skipping colocate=true, draft_size=0"
            return
        fi

        # if [ "$target_attention" = "true" ]; then
        #     return
        # fi
        $target_attention = "false"

        if [ "$demote" = "true" ]; then
            echo "Skipping demote=true, draft_size=0"
            return
        fi
    fi

    if [ "$chunk_prefill" = "false" ]; then
        # return when budget is less than 2048
        if [ "$budget" -lt 2048 ]; then
            echo "Skipping budget < 2048, chunk_prefill=false"
            return
        fi

        if [ "$demote" = "true" ]; then
            echo "Skipping demote=true, chunk_prefill=false"
            return
        fi
    fi

    chunk_prefill_flag=""
    [ "$chunk_prefill" = "true" ] && chunk_prefill_flag="--chunked-prefill"
    colocate_flag=""
    [ "$colocate" = "true" ] && colocate_flag="--colocate"
    target_attention_flag=""
    [ "$target_attention" = "true" ] && target_attention_flag="--target-attention"
    demote_spec_tokens_flag=""
    [ "$demote" = "true" ] && demote_spec_tokens_flag="--demote-spec-tokens"
    
    # save the last running python line in a file
    echo "python "$python_script" --dataset "$dataset" --input-len "$input_len" --output-len "$output_len" --request-rate "$request_rate" --draft-size "$draft_size" $chunk_prefill_flag $colocate_flag $target_attention_flag $demote_spec_tokens_flag --budget $budget --emulate-accept-prob 0.7" > last_run.sh

    local output=$(python "$python_script" --dataset "$dataset" --input-len "$input_len" --output-len "$output_len" --request-rate "$request_rate" --draft-size "$draft_size" $chunk_prefill_flag $colocate_flag $target_attention_flag $demote_spec_tokens_flag --budget $budget --emulate-accept-prob 0.7)
    
    extract_values "$output"
    echo "$dataset,$input_len,$output_len,$request_rate,$draft_size,$chunk_prefill,$colocate,$target_attention,$budget,$demote,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_tpt,$p99_tpt,$throughput,$latency,$preempt_flag" >> "$output_csv"
    ./slack "dataset: $dataset, input_len: $input_len, output_len: $output_len, request_rate: $request_rate, draft_size: $draft_size, chunk_prefill: $chunk_prefill, colocate: $colocate, attention: $target_attention, budget: $budget, demote: $demote, throughput: $throughput, latency: $latency, preempted: $preempt_flag"
}

# Initialize
initialize_csv

total_runs=$(( ${#datasets[@]} * ${#temperatures[@]} * ${#input_output_lens_pairs[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#chunk_prefills[@]} * ${#colocates[@]} * ${#target_attentions[@]} * ${#budgets[@]} * ${#demote_spec_tokens[@]}))
current_run=0

# Main Loop
for dataset in "${datasets[@]}"; do
    for temperature in "${temperatures[@]}"; do
        for budget in "${budgets[@]}"; do
            for demote in "${demote_spec_tokens[@]}"; do
                for input_output_lens_pair in "${input_output_lens_pairs[@]}"; do
                    input_len=$(echo "$input_output_lens_pair" | awk '{print $1}')
                    output_len=$(echo "$input_output_lens_pair" | awk '{print $2}')

                    for draft_size in "${draft_sizes[@]}"; do
                        for chunk_prefill in "${chunk_prefills[@]}"; do
                            for request_rate in "${request_rates[@]}"; do
                                for colocate in "${colocates[@]}"; do
                                    for target_attention in "${target_attentions[@]}"; do
                                        echo "[${current_run}/${total_runs}]"
                                        ./slack "[${current_run}/${total_runs}]"
                                        run_benchmark "$dataset" "$input_len" "$output_len" "$request_rate" "$draft_size" "$chunk_prefill" "$colocate" "$target_attention" "$budget" "$demote"
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
