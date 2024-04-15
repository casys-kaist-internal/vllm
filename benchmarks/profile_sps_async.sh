dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=4
request_rate=4
num_iters=10
num_prompts=$((request_rate * num_iters))

port=8841
delay=120
duration=150

download_dir="/home/noppanat/workspace/models"
log_file="/home/noppanat/workspace/vllm/benchmarks/logs/profile_sps_async.log"

# run server with sps engine
echo "Running sps server at port $port"

./nsys_profile \
    --delay=$delay \
    --duration=$duration \
    python3 -m vllm.entrypoints.sps_api_server \
    --port $port \
    --download-dir $download_dir \
    --target-model $target_model \
    --draft-model $draft_model \
    --disable-log-requests \
    > $log_file &
sleep 30

# warm up
./run_client.sh $port 10 > /dev/null
sleep 10

# run client
echo "Running client at port $port with $request_rate reqs/sec"

python benchmark_serving.py \
    --port $port \
    --tokenizer $target_model \
    --dataset $dataset \
    --num-prompts $num_prompts \
    --request-rate $request_rate

# kill server
# kill $(ps aux | grep '[p]ython3 -m vllm.entrypoints.sps_api_server' | awk '{print $2}')

sleep 10

echo Done
