#!/bin/bash

# Define configurations
# declare -a configs=(
#     "(12,12,64)"
#     "(16,16,64)"
#     "(32,32,64)"
#     "(32,32,80)"
#     "(32,32,128)"
#     "(40,40,128)"
#     "(56,56,128)"
# )

declare -a configs=(
    "(32,32,128)"
)

for visible_device in "0" "2" "3"; do

    echo "batch_size,context_len,num_query_heads,num_kv_heads,head_size,latency" > ${visible_devices}_v1_batch_size_output.csv
    
    # Loop over configurations
    for config in "${configs[@]}"; do
        # Remove parentheses and split by comma
        IFS=', ' read -r -a array <<< "${config//(/}"
        array[2]=${array[2]//)/}

        # Loop over batch size from 1 to 512
        for batch_size in {1..512}; do
            # Loop over context length from 1 to 4096
            for context_len in 1024; do
            # for context_len in 256 512 768 1024 1280 1536 1792 2048; do
                # Run Python script with configuration and capture the output
                output=$(CUDA_VISIBLE_DEVICES=$visible_device \
                python benchmark_paged_attention.py \
                    --version "v1" \
                    --batch-size "$batch_size" \
                    --context-len "$context_len" \
                    --num-query-heads "${array[0]}" \
                    --num-kv-heads "${array[1]}" \
                    --head-size "${array[2]}" \
                    --block-size 16 \
                    --dtype "half")

                # Extract the latency from the output
                latency=$(echo "$output" | grep -oP 'Kernel running time: \K\d+\.\d+')

                # Print the output with configuration, batch size, and context length
                echo "${batch_size},${context_len},${array[0]},${array[1]},${array[2]},${latency}" >> ${visible_devices}_v1_batch_size_output.csv
            done
        done
    done

    echo "batch_size,context_len,num_query_heads,num_kv_heads,head_size,latency" > ${visible_devices}_v1_context_len_output.csv

    # Loop over configurations
    for config in "${configs[@]}"; do
        # Remove parentheses and split by comma
        IFS=', ' read -r -a array <<< "${config//(/}"
        array[2]=${array[2]//)/}

        # Loop over batch size from 1 to 256
        for batch_size in 64; do
            # Loop over context length from 1 to 4096
            for context_len in 256 512 768 1024 1280 1536 1792 2048; do
                # Run Python script with configuration and capture the output
                output=$(CUDA_VISIBLE_DEVICES=$visible_device \
                python benchmark_paged_attention.py \
                    --version "v1" \
                    --batch-size "$batch_size" \
                    --context-len "$context_len" \
                    --num-query-heads "${array[0]}" \
                    --num-kv-heads "${array[1]}" \
                    --head-size "${array[2]}" \
                    --block-size 16 \
                    --dtype "half")

                # Extract the latency from the output
                latency=$(echo "$output" | grep -oP 'Kernel running time: \K\d+\.\d+')

                # Print the output with configuration, batch size, and context length
                echo "${batch_size},${context_len},${array[0]},${array[1]},${array[2]},${latency}" >> ${visible_devices}_v1_context_len_output.csv
            done
        done
    done

    echo "batch_size,context_len,num_query_heads,num_kv_heads,head_size,latency" > ${visible_devices}_v2_batch_size_output.csv

    # Loop over configurations
    for config in "${configs[@]}"; do
        # Remove parentheses and split by comma
        IFS=', ' read -r -a array <<< "${config//(/}"
        array[2]=${array[2]//)/}

        # Loop over batch size from 1 to 512
        for batch_size in {1..512}; do
            # Loop over context length from 1 to 4096
            for context_len in 1024; do
                # Run Python script with configuration and capture the output
                output=$(CUDA_VISIBLE_DEVICES=$visible_device \
                python benchmark_paged_attention.py \
                    --version "v2" \
                    --batch-size "$batch_size" \
                    --context-len "$context_len" \
                    --num-query-heads "${array[0]}" \
                    --num-kv-heads "${array[1]}" \
                    --head-size "${array[2]}" \
                    --block-size 16 \
                    --dtype "half")

                # Extract the latency from the output
                latency=$(echo "$output" | grep -oP 'Kernel running time: \K\d+\.\d+')

                # Print the output with configuration, batch size, and context length
                echo "${batch_size},${context_len},${array[0]},${array[1]},${array[2]},${latency}" >> ${visible_devices}_v2_batch_size_output.csv
            done
        done
    done

    echo "batch_size,context_len,num_query_heads,num_kv_heads,head_size,latency" > ${visible_devices}_v2_context_len_output.csv

    # Loop over configurations
    for config in "${configs[@]}"; do
        # Remove parentheses and split by comma
        IFS=', ' read -r -a array <<< "${config//(/}"
        array[2]=${array[2]//)/}

        # Loop over batch size from 1 to 256
        for batch_size in 64; do
            # Loop over context length from 1 to 4096
            for context_len in 256 512 768 1024 1280 1536 1792 2048; do
                # Run Python script with configuration and capture the output
                output=$(CUDA_VISIBLE_DEVICES=$visible_device \
                python benchmark_paged_attention.py \
                    --version "v2" \
                    --batch-size "$batch_size" \
                    --context-len "$context_len" \
                    --num-query-heads "${array[0]}" \
                    --num-kv-heads "${array[1]}" \
                    --head-size "${array[2]}" \
                    --block-size 16 \
                    --dtype "half")

                # Extract the latency from the output
                latency=$(echo "$output" | grep -oP 'Kernel running time: \K\d+\.\d+')

                # Print the output with configuration, batch size, and context length
                echo "${batch_size},${context_len},${array[0]},${array[1]},${array[2]},${latency}" >> ${visible_devices}_v2_context_len_output.csv
            done
        done
    done
done