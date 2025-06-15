#!/usr/bin/env bash
set -eo pipefail

export TMPDIR=""

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
NUM_WORKERS=$5
EVAL_OUTPUT_DIR=$6
CACHE_PATH=$7
DEST_PATH=$8


if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$CACHE_PATH" ]; then
  echo "Cache path not specified, exiting."
  exit 1
fi

if [ -z "$DEST_PATH" ]; then
  echo "Dest path not specified, exiting."
  exit 1
fi


get_openhands_version

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"

COMMAND="poetry run python evaluation/benchmarks/lmr_bench/run_infer.py \
  --agent-cls $AGENT \
  --llm-config $MODEL_CONFIG \
  --max-iterations 50 \
  --eval-num-workers $NUM_WORKERS \
  --eval-note $OPENHANDS_VERSION \
  --eval-output-dir $EVAL_OUTPUT_DIR \
  --cache-path $CACHE_PATH \
  --dest-path $DEST_PATH" 

# Run the command
eval $COMMAND

