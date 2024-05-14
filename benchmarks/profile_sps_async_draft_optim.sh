dataset="sharegpt"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=7
max_num_seqs=16
request_rate=$max_num_seqs
num_iters=-1
use_dynamic_draft_size=True
num_prompts=$((request_rate * num_iters))

port=8841
delay=95    # retrain_period=10000

download_dir="/home/noppanat/workspace/models"
log_file="/home/noppanat/workspace/vllm/benchmarks/logs/profile_sps_async.log"

# run server with sps engine
echo "Running sps server at port $port"

./nsys_profile \
    --delay=$delay \
    python3 -m vllm.entrypoints.sps_api_server \
    --port $port \
    --download-dir $download_dir \
    --target-model $target_model \
    --draft-model $draft_model \
    --draft-size $draft_size \
    --use-dynamic-draft-size $use_dynamic_draft_size \
    --max-num-seqs $max_num_seqs \
    --disable-log-requests \
    > $log_file &
sleep 30

# run client
echo "Running client at port $port with $request_rate reqs/sec"

python benchmark_serving_draft_optim.py \
    --port $port \
    --tokenizer $target_model \
    --dataset $dataset \
    --num-iters $num_iters \
    --request-rate $request_rate

sleep 10

echo Done
