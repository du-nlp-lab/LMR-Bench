if [ $# -ne 2 ]; then
  echo "Usage: $0 EVALUATION_DIR $1 EVALUATION_OUTPUT_DIR"
  exit 1
fi

EVALUATION_DIR=$1
EVALUATION_OUTPUT_DIR=$2

python3 evaluation/llm_as_a_judge_evaluation.py --evaluation_dir "${EVALUATION_DIR}" --evaluation_output_dir "${EVALUATION_OUTPUT_DIR}"
