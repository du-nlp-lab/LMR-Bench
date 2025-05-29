#!/bin/bash

# Define lists of parameters
MODEL_NAMES=("EleutherAI_pythia-1.4b" "Phi-1_5" "Mistral-7B-v0.1" "Meta-Llama-3-8B")
TOKENS=("first" "object" "subject_query" "relation_query")
MODULES=("--include_mlps" "--include_mlps_l1" "--include_mhsa")


MODEL_NAMES=("Meta-Llama-3-8B")
TOKENS=("first")
MODULES=("--include_mlps")

# Fixed parameters
DEVICE="cuda"
NB_COUNTER_PARAMETRIC_KNOWLEDGE="3"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for TOKEN in "${TOKENS[@]}"; do
        for MODULE in "${MODULES[@]}"; do
            COMMAND="python ./scripts/main.py --model_name ${MODEL_NAME} --device ${DEVICE} --nb_counter_parametric_knowledge ${NB_COUNTER_PARAMETRIC_KNOWLEDGE} ${MODULE} --token_position ${TOKEN}"
            
            echo "Running command: ${COMMAND}"
            eval ${COMMAND}
        done
    done
done