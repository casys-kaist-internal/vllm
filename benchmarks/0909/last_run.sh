export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

python benchmark_serving.py --dataset finance --temperature 0 --request-rate 24 --draft-size 7 --prefill-schedule-mode chunked_prefill --budget-token 64 --budget-seq 16 --target-model facebook/opt-6.7b --draft-model facebook/opt-125m 

#python benchmark_serving.py --dataset finance --temperature 0 --request-rate 24 --draft-size 7 --prefill-schedule-mode full_prefill --budget-token 4096 --budget-seq 128 --target-model facebook/opt-6.7b --draft-model facebook/opt-125m --gamma-mapping-attention 
