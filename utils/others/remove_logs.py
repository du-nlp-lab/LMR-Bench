#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

def remove_logs_in_repos(root_dir: Path):
    """
    Traverse each subfolder in root_dir, read 'repo_folder_name' from info.json,
    and delete the unit_test/logs directory in the corresponding repo folder.
    """
    for main_folder in root_dir.iterdir():
        if not main_folder.is_dir():
            continue

        info_json = main_folder / "info.json"
        if not info_json.is_file():
            print(f"⚠️ info.json not found: {main_folder}")
            continue

        try:
            # Load info.json
            data = json.loads(info_json.read_text(encoding="utf-8"))
            repo_rel = data.get("repo_folder_name")
            if not repo_rel:
                print(f"⚠️ 'repo_folder_name' field not found in info.json: {info_json}")
                continue

            # Locate the repo folder
            repo_folder = main_folder / repo_rel
            logs_dir = repo_folder / "unit_test" / "logs"

            # Remove logs directory
            if logs_dir.exists() and logs_dir.is_dir():
                shutil.rmtree(logs_dir)
                print(f"✅ Deleted: {logs_dir}")
            else:
                print(f"ℹ️ Logs directory does not exist: {logs_dir}")

        except Exception as e:
            print(f"❌ Failed to process {info_json}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Remove the unit_test/logs directory in each subproject repo"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory path containing multiple subfolders, each with an info.json file"
    )
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        parser.error(f"{args.root_dir} is not a valid directory")
    remove_logs_in_repos(args.root_dir)

if __name__ == "__main__":
    main()
