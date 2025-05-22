#!/bin/bash


export TMPDIR=/home/sxy240002/tmp
export TMP=/home/sxy240002/tmp
export TEMPDIR=/home/sxy240002/tmp
export TRANSFORMERS_CACHE=/home/sxy240002/transformers_cache
export HUGGINGFACE_HUB_TOKEN=


if [ $# -ne 1 ]; then
  echo "Usage: $0 PROJECT_DIR"
  exit 1
fi

PROJECT_DIR=$1


python3 evaluation/unit_test_evaluation_golden.py --project_dir "${PROJECT_DIR}"
