#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import json
import argparse
from pathlib import Path

def copy_and_replace(source_root: str, dest_root: str):
    """
    将 source_root 下的每个主文件夹复制到 dest_root，
    并在复制后的目录里，用 golden_file 替换 repo/goal_file。
    采用 shutil.copytree(..., copy_function=shutil.copy) 避免复制权限元数据。
    """
    # 确保目标根目录存在
    os.makedirs(dest_root, exist_ok=True)

    # 遍历每个主文件夹
    for folder_name in os.listdir(source_root):
        src_main = os.path.join(source_root, folder_name)
        if not os.path.isdir(src_main):
            continue

        dst_main = os.path.join(dest_root, folder_name)

        # 1. 复制整个主文件夹到目标位置，不复制权限元数据
        shutil.copytree(
            src_main,
            dst_main,
            dirs_exist_ok=True,
            copy_function=shutil.copy,
            ignore=shutil.ignore_patterns('.git', '.git/*')
        )
        print(f"Copied folder:\n  {src_main}\n→ {dst_main}")

        # 2. 读取 info.json
        info_path = os.path.join(dst_main, 'info.json')
        if not os.path.isfile(info_path):
            print(f"  ⚠️ 没找到 info.json，跳过：{info_path}")
            continue

        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        repo_folder = info.get("repo_folder_name", "")
        impls = info.get('implementations', [])
        if not isinstance(impls, list):
            print(f"  ⚠️ implementations 不是列表，跳过：{info_path}")
            continue

        # 3. 逐条处理 implementations
        for idx, impl in enumerate(impls, start=1):
            goal_file_rel   = impl.get('goal_file')
            golden_file_rel = impl.get('golden_file')

            if not goal_file_rel or not golden_file_rel:
                print(f"  ⚠️ implementation #{idx} 字段缺失，跳过")
                continue

            golden_src = os.path.join(dst_main, golden_file_rel)
            goal_dst   = os.path.join(dst_main, repo_folder, goal_file_rel)

            if not os.path.isfile(golden_src):
                print(f"  ⚠️ 没找到 golden_file，跳过：{golden_src}")
                continue

            os.makedirs(os.path.dirname(goal_dst), exist_ok=True)
            shutil.copy(golden_src, goal_dst)
            print(f"  Replaced:\n    {goal_dst}\n  ← {golden_src}")

def main():
    parser = argparse.ArgumentParser(
        description="批量复制主文件夹并用 golden_file 覆盖 repo/goal_file"
    )
    parser.add_argument(
        'source_root',
        help="原始主文件夹根目录（包含多个子文件夹）"
    )
    parser.add_argument(
        'dest_root',
        help="复制后放置主文件夹的新根目录"
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.source_root)
    dest_root   = os.path.abspath(args.dest_root)

    print(f"Source: {source_root}\nDestination: {dest_root}\n")
    copy_and_replace(source_root, dest_root)

if __name__ == '__main__':
    main()