#!/usr/bin/env bash

export OPENAI_API_KEY="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"
export OPENAI_API_BASE="https://api.anthropic.com/v1/"

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 generation/base_agent_generation_openai.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"
