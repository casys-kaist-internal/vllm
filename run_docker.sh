#!/bin/bash
workspace="/home/hjlee/workspace"
data="/data"

docker run --gpus '"device=2"' -it --rm \
--ipc host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--shm-size=12g -p 8003:8003 \
-v $workspace:$workspace \
-v $data:$data \
-w $workspace \
sps_vllm_hj