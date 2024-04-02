#!/bin/bash

# The directory containing your Docker command and API server code
directory="."

# Change to the correct directory
cd $directory

docker build -t speculative_decoding_hj_multi -f ./Dockerfile . --no-cache