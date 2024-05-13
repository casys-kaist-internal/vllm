dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=4
batch_size=4
num_iters=3

delay=16
duration=6

./nsys_profile \
    --delay=$delay \
    --duration=$duration \
    python benchmark_sps_latency.py \
    --engine sps \
    --dataset $dataset \
    --target-model $target_model \
    --draft-model $draft_model \
    --draft-size $draft_size \
    --batch-size $batch_size \
    --num-iters $num_iters