#!/bin/bash

ncu -k regex:paged_attention_v2_ --set full -f -o attention \
python3 bench_hj.py \
--batch-size 32 \
--query-len 4 \
--context-len 2048
