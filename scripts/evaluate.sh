#!/bin/bash

export LLM_API_KEY=""
export LLM_MODEL="openai/o3-mini"

EVALUATION_DIR=$1
EVALUATION_OUTPUT_DIR=$2

for subdir in "$EVALUATION_DIR"/*/; do
    echo "Processing subdirectory: $subdir"

    script="${subdir}evaluate.sh"
    
    # Check if the file exists and is a regular file.
    if [ -f "$script" ]; then
        echo "Executing $script"
        bash "$script"  # Or sh "$script" or . "$script" if you want to source it
    else
        echo "No evaluate.sh found in $subdir"
    fi
done
