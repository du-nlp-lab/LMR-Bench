#!/bin/bash

export LLM_API_KEY="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"
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
