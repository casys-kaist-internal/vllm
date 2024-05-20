#!/bin/bash

args=("$@")
port=${args[0]}

# check that port is a valid integer of length 4
if ! [[ "$port" =~ ^[0-9]+$ ]] || [ ${#port} -ne 4 ]; then
    echo "Invalid port: $port"
    exit 1
fi

# set result csv file 
output_file="/home/noppanat/workspace/vllm/benchmarks/results_serving.csv"
log_file="/home/noppanat/workspace/vllm/benchmarks/logs/run_serving_debug.log"
rm $output_file

# draft size
draft_size=2

# request rate list 
request_rate_list=(4 8 16 32 64 inf)

echo "Engine, Request rate (req/s), Total time (s), Throughput (req/s), Average latency (s), Average latency per token (s), Average latency per output token (s)" >> $output_file

# run server with sps engine
./run_server.sh sps $port $draft_size > $log_file &
sleep 30

# warm up
./run_client.sh $port 10 > /dev/null
sleep 10

# run client with sps engine sweeping request rate list
for request_rate in ${request_rate_list[@]}; do
    echo -n "sps_$draft_size, $request_rate, " >> $output_file
    ./run_client.sh $port $request_rate \
    | awk -F': ' '{print $2}' \
    | grep -o '[0-9]\+\.[0-9]\+' \
    | paste -sd, >> $output_file
    echo "SpS $request_rate Done"
    sleep 10
done

# kill server
kill $(ps aux | grep '[p]ython3 -m vllm.entrypoints.sps_api_server' | awk '{print $2}')

echo "SpS $draft_size Done"
sleep 10

echo "Finished"