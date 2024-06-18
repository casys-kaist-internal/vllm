#!/bin/bash
PROFILE=false

# Default values
DOWNLOAD_DIR='/mnt/sda/download'
TARGET_MODEL='facebook/opt-6.7b'
DRAFT_MODEL='facebook/opt-125m'
DRAFT_SIZE=4
COLLOCATE=false
CHUNKED_PREFILL=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --profile) PROFILE=true ;;
        -c) COLLOCATE=true ;;
        -cp) CHUNKED_PREFILL=true ;;
        --target-model) TARGET_MODEL="$2"; shift ;;
        --draft-model) DRAFT_MODEL="$2"; shift ;;
        --draft-size) DRAFT_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Build the command
COMMAND="python -m vllm.entrypoints.spec_decode_api_server \
        --target-model $TARGET_MODEL \
        --draft-model $DRAFT_MODEL \
        --draft-size $DRAFT_SIZE \
        --swap-space 16 \
        --download-dir $DOWNLOAD_DIR \
        --disable-log-requests"

# Add profile option if specified
if [ "$PROFILE" = true ]; then
    COMMAND="./nsys_profile.sh $COMMAND"
fi

# Add collocate option if specified
if [ "$COLLOCATE" = true ]; then
    COMMAND="$COMMAND --collocate"
fi

# Add chunked prefill option if specified
if [ "$CHUNKED_PREFILL" = true ]; then
    COMMAND="$COMMAND --enable-chunked-prefill"
fi

# Execute the command
eval $COMMAND