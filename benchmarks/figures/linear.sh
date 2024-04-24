#!/bin/bash
# Run python linear.py <input_features> <output_features>

for visible_device in "0" "2" "3"; do
    input_features=4096
    output_features=4096

    CUDA_VISIBLE_DEVICES=$visible_device python linear.py $input_features $output_features

    input_features=4096
    output_features=$((4096*3))

    CUDA_VISIBLE_DEVICES=$visible_device python linear.py $input_features $output_features

    input_features=4096
    output_features=$((4096*4))

    CUDA_VISIBLE_DEVICES=$visible_device python linear.py $input_features $output_features

    input_features=$((4096*4))
    output_features=4096

    CUDA_VISIBLE_DEVICES=$visible_device python linear.py $input_features $output_features
done