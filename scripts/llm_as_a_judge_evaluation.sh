export OPENAI_API_KEY=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

if [ $# -ne 2 ]; then
  echo "Usage: $0 EVALUATION_DIR $1 EVALUATION_OUTPUT_DIR"
  exit 1
fi

EVALUATION_DIR=$1
EVALUATION_OUTPUT_DIR=$2

python3 evaluation/llm_as_a_judge_evaluation.py --evaluation_dir "${EVALUATION_DIR}" --evaluation_output_dir "${EVALUATION_OUTPUT_DIR}"
