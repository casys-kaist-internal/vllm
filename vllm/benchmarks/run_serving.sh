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
rm $output_file

# request rate list 
request_rate_list=(4 8 16 32 64 inf)

echo "Engine, Request rate (req/s), Total time (s), Throughput (req/s), Average latency (s), Average latency per token (s), Average latency per output token (s)" >> $output_file

./run_server.sh base $port &

# wait for server to be ready
sleep 30

# warm up
./run_client.sh $port 10 > /dev/null
sleep 10

# run client with base engine sweeping request rate list
for request_rate in ${request_rate_list[@]}; do
    echo -n "base, $request_rate, " >> $output_file
    ./run_client.sh $port $request_rate \
    | awk -F': ' '{print $2}' \
    | grep -o '[0-9]\+\.[0-9]\+' \
    | paste -sd, >> $output_file
    slack "Base $request_rate Done"
    sleep 10
done

# kill server 
kill $(ps aux | grep '[p]ython3 -m vllm.entrypoints.api_server' | awk '{print $2}')

slack "Base Done"

# Loop through draft size from 2 to 8
for draft_size in {2..4}; do
    # run server with sps engine
    ./run_server.sh sps $port $draft_size &
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
        slack "SpS $request_rate Done"
        sleep 10
    done

    # kill server
    kill $(ps aux | grep '[p]ython3 -m vllm.entrypoints.sps_api_server' | awk '{print $2}')

    slack "SpS $draft_size Done"
    sleep 10
done

slack "Finished"