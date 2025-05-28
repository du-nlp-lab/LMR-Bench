#!/bin/bash

export TMPDIR=/home/sxy240002/tmp
export TMP=/home/sxy240002/tmp
export TEMPDIR=/home/sxy240002/tmp
export TRANSFORMERS_CACHE=/home/sxy240002/transformers_cache
export HUGGINGFACE_HUB_TOKEN=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


if [ $# -ne 2 ]; then
  echo "Usage: $0 EVALUATION_DIR $1 EVALUATION_OUTPUT_DIR"
  exit 1
fi

EVALUATION_DIR=$1
EVALUATION_OUTPUT_DIR=$2

python3 evaluation/unit_test_evaluation.py --evaluation_dir "${EVALUATION_DIR}" --evaluation_output_dir "${EVALUATION_OUTPUT_DIR}"
# python3 evaluation/unit_test_evaluation.py --evaluation_dir "${EVALUATION_DIR}"