#!/bin/bash

DOWNLOAD_DIR='/mnt/sda/download'
TARGET_MODEL='facebook/opt-6.7b'
DRAFT_MODEL='facebook/opt-125m'
DRAFT_SIZE=7

python -m vllm.entrypoints.spec_decode_api_server \
        --target-model $TARGET_MODEL \
        --draft-model $DRAFT_MODEL \
        --draft-size $DRAFT_SIZE \
        --swap-space 16 \
        --download-dir $DOWNLOAD_DIR \
        --disable-log-requests