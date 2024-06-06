#!/bin/bash

mkdir -p nsight
OUTFILE='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')
nsys profile -t cuda,cudnn,cublas,nvtx --gpu-metrics-device=0 --cuda-graph-trace=node --output=${OUTFILE} $@
