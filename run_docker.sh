docker run -it \
  --gpus '"device=3"' \
  --rm -p 8000:8000 --shm-size=12g \
  -v .:/workspace/vllm \
  speculative_decoding_hj \
  /bin/bash