#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

def remove_logs_in_repos(root_dir: Path):
    """
    遍历 root_dir 下的每个子文件夹，读取 info.json 中的 repo_folder_name，
    然后删除对应 repo_folder 下的 unit_test/logs 目录。
    """
    for main_folder in root_dir.iterdir():
        if not main_folder.is_dir():
            continue

        info_json = main_folder / "info.json"
        if not info_json.is_file():
            print(f"⚠️ 未找到 info.json: {main_folder}")
            continue

        try:
            # 读取 info.json
            data = json.loads(info_json.read_text(encoding="utf-8"))
            repo_rel = data.get("repo_folder_name")
            if not repo_rel:
                print(f"⚠️ info.json 中没有字段 repo_folder_name: {info_json}")
                continue

            # 定位到 repo 文件夹
            repo_folder = main_folder / repo_rel
            logs_dir = repo_folder / "unit_test" / "logs"

            # 删除 logs 目录
            if logs_dir.exists() and logs_dir.is_dir():
                shutil.rmtree(logs_dir)
                print(f"✅ 已删除: {logs_dir}")
            else:
                print(f"ℹ️ logs 目录不存在: {logs_dir}")

        except Exception as e:
            print(f"❌ 处理失败 {info_json}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="删除每个子项目 repo 下 unit_test/logs 目录"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="主目录路径，里面包含多个子文件夹，每个子文件夹下有 info.json"
    )
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        parser.error(f"{args.root_dir} 不是一个有效的目录")
    remove_logs_in_repos(args.root_dir)

if __name__ == "__main__":
    main()
