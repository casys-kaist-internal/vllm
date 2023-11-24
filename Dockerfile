# Start from the NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install tmux -y

# Set timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN apt-get install -y tzdata

# Set the working directory
WORKDIR /workspace/vllm

# Upgrade pip
RUN pip install --upgrade pip
RUN pip uninstall torch -y

# Try to install the vllm Python package twice if the first attempt fails