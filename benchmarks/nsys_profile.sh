#!/bin/bash

# first argument is batch size

# erase previous profile files
rm -r -f profile_nsys
mkdir -p profile_nsys

for BATCH_SIZE in 127 128 129
do
    echo "batch size: ${BATCH_SIZE}"
    OUTFILE='profile_nsys/'$(date '+%Y-%m-%d_%H-%M-%S')'-test-'${BATCH_SIZE}
    nsys profile --gpu-metrics-device=all --delay=1 --trace=nvtx,cuda --gpu-metrics-frequency=100000 --output=${OUTFILE} python3 profile.py --fixed-batch-size=${BATCH_SIZE}
done
