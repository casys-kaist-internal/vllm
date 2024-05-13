dataset="sharegpt"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=7
batch_size=16
num_iters=-1

delay=80    # retrain_period=10000

./nsys_profile \
    --delay=$delay \
    python benchmark_sps_latency_draft_optim.py \
    --engine sps \
    --dataset $dataset \
    --target-model $target_model \
    --draft-model $draft_model \
    --draft-size $draft_size \
    --batch-size $batch_size \
    --num-iters $num_iters \
    --dynamic-draft