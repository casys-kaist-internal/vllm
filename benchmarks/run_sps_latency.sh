dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=4
# batch_size=4
num_iters=3

log_file="logs/run_sps_latency.log"
rm $log_file > /dev/null

for batch_size in 1 2 4 8 16; do
    python benchmark_sps_latency.py \
        --engine sps \
        --dataset $dataset \
        --target-model $target_model \
        --draft-model $draft_model \
        --draft-size $draft_size \
        --batch-size $batch_size \
        --num-iters $num_iters >> $log_file
done