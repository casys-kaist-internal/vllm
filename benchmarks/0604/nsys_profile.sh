#!/bin/bash

mkdir -p nsight
CURRENT_DATE=$(date '+%Y-%m-%d_%H-%M')
OUTFILE='nsight/'${CURRENT_DATE}
nsys profile -t cuda,cudnn,cublas,nvtx --gpu-metrics-device=0 --cuda-graph-trace=node --output=${OUTFILE} $@
RAY_OUTFILE='nsight/'${CURRENT_DATE}'_ray.nsys-rep'
mv /tmp/ray/session_latest/logs/nsight/* ${RAY_OUTFILE}