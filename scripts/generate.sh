#!/bin/bash

export LLM_API_KEY=""
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