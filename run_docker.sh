docker run -it \
  --gpus '"device=0"' \
  --rm -p 8001:8001 --shm-size=12g \
  --cap-add SYS_ADMIN \
  -v .:/workspace/vllm \
  speculative_decoding_hj_latest \
  /bin/bash