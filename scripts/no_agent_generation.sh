#!/usr/bin/env bash

export OPENAI_API_KEY=""

# Parse positional arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
MODEL="$3"

# Validate arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL" ]; then
  echo "Usage: $0 <input_dir> <output_dir> <model>"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python generation script
python3 generation/noagent/base_agent_generation_openai.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model "$MODEL"