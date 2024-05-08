dataset="/home/noppanat/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
target_model="facebook/opt-6.7b"
draft_model="facebook/opt-125m"
draft_size=4
request_rate=4
num_iters=3
num_prompts=$((request_rate * num_iters))

port=8841

download_dir="/home/noppanat/workspace/models"
log_file="logs/run_sps_serving.log"
rm $log_file > /dev/null

python3 -m vllm.entrypoints.sps_api_server \
    --port $port \
    --download-dir $download_dir \
    --target-model $target_model \
    --draft-model $draft_model \
    --draft-size $draft_size \
    --disable-log-requests \
    >> $log_file 2>> $log_file &

# warm up
./run_client.sh $port 10 > /dev/null
sleep 30

# run client
echo "Running client at port $port with $request_rate reqs/sec"

python benchmark_serving.py \
    --port $port \
    --tokenizer $target_model \
    --dataset $dataset \
    --num-prompts $num_prompts \
    --request-rate $request_rate

kill $(ps aux | grep '[p]ython3 -m vllm.entrypoints.api_server' | awk '{print $2}')

sleep 10

echo Done