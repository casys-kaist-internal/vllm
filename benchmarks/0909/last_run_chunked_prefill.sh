python benchmark_serving.py --dataset sharegpt --temperature -1 --request-rate 16 --draft-size 7 --prefill-schedule-mode chunked_prefill --budget-token 2048 --budget-seq 128 --colocate --gamma-mapping-attention --drop-threshold 0.25 --target-model facebook/opt-6.7b --draft-model facebook/opt-125m
