#!/bin/bash

### how to use ###
# ncu_profile.sh {KERNERL_NAME} {EXTRA ARGUMENT FOR NCU} {EXECUTABLE FILE}
# By default, the value of c is set to 1, but it can be changed by redefining it in the {EXTRA ARGUMENT}.
#
# i.e) ncu_profile.sh \
#      single_query_cached_kv_attention_kernel \
#      -c 2 \
#      python vllm/benchmarks/benchmark_throughput.py --backend vllm --dataset vllm/ShareGPT_V3_unfiltered_cleaned_split.json --model facebook/opt-125m  --num-prompts=1000 --tensor-parallel-size=1


# output path
mkdir -p profile_ncu
OUTFILE='profile_ncu/'$(date '+%Y-%m-%d_%H-%M-%S')

# kerrnerl name
KERNEL_NAME="$1"
shift

# c default
C_VALUE=1

# -c option
for i in "$@"; do
  if [[ "$i" == "-c" ]]; then
    C_FLAG=true
  elif [[ "$C_FLAG" == true ]]; then
    C_VALUE="$i"
    C_FLAG=false 
    break
  fi
done

# -c option delete
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "-c" ]]; then
    SKIP_NEXT=true
  elif [[ "$SKIP_NEXT" == true ]]; then
    SKIP_NEXT=false
    continue
  else
    ARGS+=("$arg")
  fi
done

# CMD
CMD="ncu -k ${KERNEL_NAME} -s 10 -c ${C_VALUE} --target-processes all --devices 0 --import-source yes --resolve-source-file "/workspace/vllm/csrc" --set full -o ${OUTFILE}  ${ARGS[@]}"
# CMD="ncu -s 10 -c ${C_VALUE} --target-processes all --devices 0 --import-source yes --resolve-source-file "/workspace/vllm/csrc" --set full -o ${OUTFILE}  ${ARGS[@]}"

echo $CMD

# eval CMD
eval $CMD

# scp to local
# scp -rp "profile_ncu" hjlee@143.248.139.14:/Users/hjlee/Desktop/nsight
# rm -r -f profile_ncu
