#!/bin/bash

declare -a models=(
    # "facebook/opt-6.7b,facebook/opt-125m"
    # "facebook/opt-6.7b,facebook/opt-350m"
    # "facebook/opt-13b,facebook/opt-125m"
    # "facebook/opt-13b,facebook/opt-350m"
    # "daryl149/llama-2-7b-chat-hf,Felladrin/Llama-68M-Chat-v1"
    # "bigscience/bloom-7b1,bigscience/bloomz-560m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-12b,EleutherAI/pythia-31m"
    "EleutherAI/pythia-12b,EleutherAI/pythia-14m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-410m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-160m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-70m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-31m"
    # "EleutherAI/pythia-6.9b,EleutherAI/pythia-14m"
)

datasets=("alpaca")
temperatures=(0.0 0.5 1.0)
batch_sizes=(1 5 10 15 20 25 30 35 40 45 50 55 60 65)
agg_types=("median")
output_len=(1024 512)

# Calculate total number of iterations
total_num_progress=$(( ${#models[@]} * ${#datasets[@]} * ${#temperatures[@]} * ${#batch_sizes[@]} * ${#agg_types[@]} * ${#output_len[@]}))
current_progress=0

# Start time
start_time=$(date +%s)

for output_len in "${output_len[@]}"; do
    for agg_type in "${agg_types[@]}"; do
        for model_pair in "${models[@]}"; do
            IFS=',' read -r target_model draft_model <<< "$model_pair"
            for batch_size in "${batch_sizes[@]}"; do
                for dataset in "${datasets[@]}"; do
                    for temp in "${temperatures[@]}"; do 
                        python3 5_17.py \
                            --target-model "$target_model" \
                            --draft-model "$draft_model" \
                            --dataset "$dataset" \
                            --temperature "$temp" \
                            --batch-size "$batch_size" \
                            --use-lookup-table \
                            --predictor-agg-type "$agg_type" \
                            --output-len "$output_len"
                        
                        # Increment current progress
                        ((current_progress++))
                        sleep 1
                        
                         # Calculate and print progress percentage
                        progress_percentage=$(echo "scale=2; ($current_progress / $total_num_progress) * 100" | bc)
                        
                        # Calculate elapsed time and estimated time left
                        current_time=$(date +%s)
                        elapsed_time=$((current_time - start_time))
                        avg_time_per_iter=$(echo "scale=2; $elapsed_time / $current_progress" | bc)
                        estimated_time_left=$(echo "scale=2; ($total_num_progress - $current_progress) * $avg_time_per_iter" | bc)
                        
                        # Convert estimated time left to integer values for hours, minutes, and seconds
                        hours=$(echo "$estimated_time_left/3600" | bc)
                        minutes=$(echo "($estimated_time_left%3600)/60" | bc)
                        seconds=$(echo "$estimated_time_left%60" | bc)

                        # force integer values
                        hours=${hours%.*}
                        minutes=${minutes%.*}
                        seconds=${seconds%.*}
                        
                        # Format estimated time left as HH:MM:SS
                        estimated_time_left_formatted=$(printf '%02d:%02d:%02d' $hours $minutes $seconds)
                        
                        echo "Progress: $current_progress / $total_num_progress ($progress_percentage%)"
                        echo "Estimated time left: $estimated_time_left_formatted"
                        ./slack "Progress: $current_progress / $total_num_progress ($progress_percentage%) - Estimated time left: $estimated_time_left_formatted"
                    done
                done
            done
        done
    done
done