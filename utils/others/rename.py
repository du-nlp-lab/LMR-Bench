#!/usr/bin/env python3
import json
import os
import sys

# List of model names to process
model_names = ['gpt4.1', 'o4mini', 'gpt4o']

# Mapping from (model_name, session_uuid) to paper_name
uuid_to_paper = {}

# First pass: build mapping from UUIDs to paper folder names
for model in model_names:
    sessions_root = f'/home/sxy240002/research_agent/OpenHands/evaluation/benchmarks/nlpbench/outputs/{model}/file_store/sessions'
    for session_uuid in os.listdir(sessions_root):
        cache_dir = os.path.join(sessions_root, session_uuid, 'event_cache')
        if not os.path.isdir(cache_dir):
            continue

        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                events = json.load(f)

            for event in events:
                if event.get('action') == 'read':
                    path_parts = event['args']['path'].split('/')
                    # Expect at least .../datasets/<paper_name>/...
                    if len(path_parts) > 4 and path_parts[3] == 'datasets':
                        paper = path_parts[4]
                        key = (model, session_uuid)
                        if key not in uuid_to_paper:
                            uuid_to_paper[key] = paper
                        # Once we've found the paper name in this session, move on
                        break

# Second pass: rename session folders using the paper names
base_root = '/home/sxy240002/research_agent/NLPAgentBench/OpenHands_logs'

for model in model_names:
    sessions_dir = os.path.join(base_root, model, 'file_store', 'sessions')
    if not os.path.isdir(sessions_dir):
        print(f"Skipping: directory not found: {sessions_dir}", file=sys.stderr)
        continue

    for old_uuid in os.listdir(sessions_dir):
        old_path = os.path.join(sessions_dir, old_uuid)
        if not os.path.isdir(old_path):
            # Skip any non-directory entries
            continue

        key = (model, old_uuid)
        if key not in uuid_to_paper:
            print(f"Warning: no mapping found for {key}", file=sys.stderr)
            continue

        new_name = uuid_to_paper[key]
        new_path = os.path.join(sessions_dir, new_name)

        # Avoid overwriting an existing folder
        if os.path.exists(new_path):
            print(f"Target already exists, skipping rename: {new_path}", file=sys.stderr)
            continue

        try:
            os.rename(old_path, new_path)
            print(f"Successfully renamed: {old_path} → {new_path}")
        except Exception as e:
            print(f"Failed to rename {old_path} → {new_path}: {e}", file=sys.stderr)
