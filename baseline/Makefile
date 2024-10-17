default: build

help:
	@echo 'Management commands for vllm_sjlim:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the vllm_sjlim project.'
	@echo '    make preprocess       Preprocess step.'
	@echo '    make run              Boot up Docker container.'
	@echo '    make up               Build and run the project.'
	@echo '    make rm               Remove Docker container.'
	@echo '    make stop             Stop Docker container.'
	@echo '    make reset            Stop and remove Docker container.'
	@echo '    make docker-setup     Setup Docker permissions for the user.'

preprocess:
	@echo "Running preprocess step"
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t vllm_sjlim

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus all --ipc=host --name vllm_sjlim -v `pwd`:/workspace vllm_sjlim:latest /bin/bash

up: build run

rm: 
	@echo "Removing Docker container"
	@docker rm vllm_sjlim

stop:
	@echo "Stopping Docker container"
	@docker stop vllm_sjlim

reset: stop rm

docker-setup:
	@echo "Setting up Docker permissions for the current user"
	@sudo groupadd docker || true
	@sudo usermod -aG docker $(USER)
	@newgrp docker
