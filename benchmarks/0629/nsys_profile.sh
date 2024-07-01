#!/bin/bash

mkdir -p nsight
CURRENT_DATE=$(date '+%Y-%m-%d_%H-%M')
OUTFILE='nsight/'${CURRENT_DATE}
nsys profile -t cuda,cudnn,cublas,nvtx --gpu-metrics-device=0 --cuda-graph-trace=node --output=${OUTFILE} $@