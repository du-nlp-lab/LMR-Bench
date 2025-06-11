#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
copy_and_replace.py

This script copies each main folder under source_root to dest_root,
and in the copied directory, replaces the repo/goal_file with the golden_file.
It uses shutil.copytree(..., copy_function=shutil.copy) to avoid copying permission metadata.
"""

import os
import shutil
import json
import argparse
from pathlib import Path

def copy_and_replace(source_root: str, dest_root: str):
    """
    Copy each main folder from source_root to dest_root,
    and replace repo/goal_file with the golden_file in the copied directory.
    Uses shutil.copytree(..., copy_function=shutil.copy) to avoid copying permission metadata.
    """
    # Ensure the destination root directory exists
    os.makedirs(dest_root, exist_ok=True)

    # Iterate through each main folder
    for folder_name in os.listdir(source_root):
        src_main = os.path.join(source_root, folder_name)
        if not os.path.isdir(src_main):
            continue

        dst_main = os.path.join(dest_root, folder_name)

        # 1. Copy the entire main folder to the destination without copying permission metadata
        shutil.copytree(
            src_main,
            dst_main,
            dirs_exist_ok=True,
            copy_function=shutil.copy,
            ignore=shutil.ignore_patterns('.git', '.git/*')
        )
        print(f"Copied folder:\n  {src_main}\n→ {dst_main}")

        # 2. Read info.json
        info_path = os.path.join(dst_main, 'info.json')
        if not os.path.isfile(info_path):
            print(f"  ⚠️ info.json not found, skipping: {info_path}")
            continue

        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        repo_folder = info.get("repo_folder_name", "")
        impls = info.get('implementations', [])
        if not isinstance(impls, list):
            print(f"  ⚠️ implementations is not a list, skipping: {info_path}")
            continue

        # 3. Process each implementation entry
        for idx, impl in enumerate(impls, start=1):
            goal_file_rel   = impl.get('goal_file')
            golden_file_rel = impl.get('golden_file')

            if not goal_file_rel or not golden_file_rel:
                print(f"  ⚠️ implementation #{idx} missing fields, skipping")
                continue

            golden_src = os.path.join(dst_main, golden_file_rel)
            goal_dst   = os.path.join(dst_main, repo_folder, goal_file_rel)

            if not os.path.isfile(golden_src):
                print(f"  ⚠️ golden_file not found, skipping: {golden_src}")
                continue

            os.makedirs(os.path.dirname(goal_dst), exist_ok=True)
            shutil.copy(golden_src, goal_dst)
            print(f"  Replaced:\n    {goal_dst}\n  ← {golden_src}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch copy main folders and replace repo/goal_file with golden_file"
    )
    parser.add_argument(
        'source_root',
        help="Root directory containing original main folders (multiple subfolders)"
    )
    parser.add_argument(
        'dest_root',
        help="New root directory for placing copied main folders"
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.source_root)
    dest_root   = os.path.abspath(args.dest_root)

    print(f"Source: {source_root}\nDestination: {dest_root}\n")
    copy_and_replace(source_root, dest_root)

if __name__ == '__main__':
    main()