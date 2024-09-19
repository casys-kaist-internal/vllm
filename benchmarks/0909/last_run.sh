export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

python benchmark_serving.py --dataset finance --temperature 0 --request-rate 24 --draft-size 7 --prefill-schedule-mode full_prefill --budget-token 4096 --budget-seq 64 --target-model facebook/opt-13b --draft-model facebook/opt-350m --gamma-mapping-attention --selective-validation --drop-threshold 0.3 --colocate
