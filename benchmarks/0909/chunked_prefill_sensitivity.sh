#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

declare -a models=(
    # "lmsys/vicuna-7b-v1.5,JackFram/llama-160m"
    "facebook/opt-6.7b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-350m"
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-13b,facebook/opt-350m"
    # "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "bigscience/bloom-7b1,bigscience/bloomz-560m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-14m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)

# Configurations
datasets=("gsm8k")
temperatures=(-1)
request_rates=(16)
draft_sizes=(7)
prefill_schedule_modes=("chunked_prefill")
budget_tokens=(256)
budget_seqs=(256)
colocates=(true)
gamma_mapping_attentions=(true)
drop_thresholds=(0 0.25)

# Paths
python_script="benchmark_serving.py"
output_csv="figures/chunked_prefill.csv"

# make directory if not exists
mkdir -p figures

initialize_csv() {
    if [ ! -f "$output_csv" ]; then
        echo "target_model,draft_model,dataset,temperature,request_rate,draft_size,prefill_schedule_mode,budget_token,budget_seq,colocate,target_attention,drop_threshold,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_token_latency,p99_token_latency,token_throughput,request_throughput,token_latency,preempt_flag" > "$output_csv"
    else
        echo "Warning: $output_csv already exists."
    fi
}

# Function to extract values from the benchmark output
extract_values() {
    local output="$1"
    request_throughput=$(echo "$output" | awk -F', ' '/result/{print $2}')
    token_throughput=$(echo "$output" | awk -F', ' '/result/{print $3}')
    token_latency=$(echo "$output" | awk -F', ' '/result/{print $4}')
    p50_ttft=$(echo "$output" | awk -F', ' '/result/{print $5}')
    p99_ttft=$(echo "$output" | awk -F', ' '/result/{print $6}')
    p50_tpot=$(echo "$output" | awk -F', ' '/result/{print $7}')
    p99_tpot=$(echo "$output" | awk -F', ' '/result/{print $8}')
    p50_token_latency=$(echo "$output" | awk -F', ' '/result/{print $9}')
    p99_token_latency=$(echo "$output" | awk -F', ' '/result/{print $10}')
    preempt_flag=$(echo "$output" | awk -F', ' '/result/{print $11}')
}

# Run the benchmark for the current configuration
run_benchmark() {
    local dataset="$1"
    local temperature="$2"
    local request_rate="$3"
    local draft_size="$4"   
    local prefill_schedule_mode="$5"
    local budget_token="$6"
    local budget_seq="$7"
    local colocate="$8"
    local gamma_mapping_attention="$9"
    local drop_threshold="${10}"
    local target_model="${11}"
    local draft_model="${12}"

    if [ "$draft_size" = "0" ]; then
        gamma_mapping_attention="false"
        colocate="false"
        # skip if colocate = true and target_attention = true only when draft_size > 0
        # if [ "$colocate" = "true" ];then
        #     return
        # fi
        # skip if drop_threshold > 0
        if [ "$drop_threshold" != "0" ];then
            return
        fi
    fi

    echo "Running benchmark for $dataset, temperature: $temperature, request_rate: $request_rate, draft_size: $draft_size, prefill_schedule_mode: $prefill_schedule_mode, budget_token: $budget_token, budget_seq: $budget_seq, colocate: $colocate, attention: $gamma_mapping_attention, drop_threshold: $drop_threshold, target_model: $target_model, draft_model: $draft_model"

    colocate_flag=""
    [ "$colocate" = "true" ] && colocate_flag="--colocate"
    gamma_mapping_attention_flag=""
    [ "$gamma_mapping_attention" = "true" ] && gamma_mapping_attention_flag="--gamma-mapping-attention"

    selective_validation_flag=""
    [ "$drop_threshold" != "0" ] && selective_validation_flag="--selective-validation"

    # save the last running python line in a file
    echo "python "$python_script" --dataset "$dataset" --temperature "$temperature" --request-rate "$request_rate" --draft-size "$draft_size" --prefill-schedule-mode "$prefill_schedule_mode" --budget-token $budget_token --budget-seq $budget_seq $colocate_flag $gamma_mapping_attention_flag $selective_validation_flag --drop-threshold $drop_threshold --target-model "$target_model" --draft-model "$draft_model"" > last_run.sh
    cat last_run.sh

    local output=$(python "$python_script" --dataset "$dataset" --temperature "$temperature" --request-rate "$request_rate" --draft-size "$draft_size" --prefill-schedule-mode "$prefill_schedule_mode" --budget-token $budget_token --budget-seq $budget_seq $colocate_flag $gamma_mapping_attention_flag $selective_validation_flag --drop-threshold $drop_threshold --target-model "$target_model" --draft-model "$draft_model")
    
    extract_values "$output"
    echo "$target_model,$draft_model,$dataset,$temperature,$request_rate,$draft_size,$prefill_schedule_mode,$budget_token,$budget_seq,$colocate,$gamma_mapping_attention,$drop_threshold,$p50_ttft,$p99_ttft,$p50_tpot,$p99_tpot,$p50_token_latency,$p99_token_latency,$token_throughput,$request_throughput,$token_latency,$preempt_flag" >> "$output_csv"
}

# Initialize
initialize_csv

total_runs=$(( ${#datasets[@]} * ${#temperatures[@]} * ${#request_rates[@]} * ${#draft_sizes[@]} * ${#prefill_schedule_modes[@]} * ${#budget_tokens[@]} * ${#budget_seqs[@]} * ${#colocates[@]} * ${#target_attentions[@]} * ${#drop_thresholds[@]} * ${#models[@]} ))
current_run=0

# Main Loop
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
                                    for gamma_mapping_attention in "${gamma_mapping_attentions[@]}"; do
                                        for drop_threshold in "${drop_thresholds[@]}"; do
                                            echo "[${current_run}/${total_runs}]"
                                            run_benchmark "$dataset" "$temperature" "$request_rate" "$draft_size" "$prefill_schedule_mode" "$budget_token" "$budget_seq" "$colocate" "$gamma_mapping_attention" "$drop_threshold" "$target_model" "$draft_model"
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
