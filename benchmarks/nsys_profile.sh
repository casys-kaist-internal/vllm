#!/bin/bash

mkdir -p nsight
OUTFILE='nsight/'$(date '+%Y-%m-%d_%H-%M-%S')
nsys profile -t cuda,osrt,nvtx,cudnn,cublas --gpu-metrics-device=0 --output=${OUTFILE} --export=${OUTFILE} $@
