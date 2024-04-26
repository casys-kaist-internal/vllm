docker run -it \
  --gpus '"device=2"' \
  --rm -p 8012:8012 --shm-size=12g \
  --cap-add SYS_ADMIN \
  -v .:/workspace/vllm \
  speculative_decoding_hj_multi \
  /bin/bash