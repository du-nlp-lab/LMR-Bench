#!/bin/bash

export LLM_API_KEY="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"
export LLM_MODEL="openai/o3-mini"

dataset_dir="/home/sxy240002/research_agent/NLPBench/benchmark/datasets"

subdirs=( "$dataset_dir"/*/ )
total=${#subdirs[@]}
count=0

for subdir in "${subdirs[@]}"; do
    count=$((count + 1))

    percent=$(( count * 100 / total ))
    printf "\nProcessing subdirectory: %s [%d/%d, %d%%]\n" "$subdir" "$count" "$total" "$percent"
    
    script="${subdir}generate.sh"



    if [ -f "$script" ]; then
        echo "Executing $script"
        bash "$script" 
    else
        echo "No generate.sh found in $subdir"
    fi
done