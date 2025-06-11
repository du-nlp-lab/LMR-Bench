#!/usr/bin/env python3
"""
copy_and_rename_outputs.py

Usage:
    python copy_and_rename_outputs.py /path/to/root_directory

This script copies the `outputs` directory inside the specified root directory
to a new directory named `outputs_renamed` alongside it.
"""

import os
import shutil
import sys
import argparse

def copy_and_rename_outputs(src_root: str):
    """
    Copy everything under src_root/outputs into a new directory named
    outputs_renamed in the same parent directory.
    """
    # define source and destination paths
    src_dir = os.path.join(src_root, "outputs")
    dst_dir = os.path.join(src_root, "outputs_renamed")

    # check that the source directory exists
    if not os.path.isdir(src_dir):
        print(f"Error: source directory does not exist → {src_dir}", file=sys.stderr)
        sys.exit(1)

    # ensure the destination does not already exist to avoid overwriting
    if os.path.exists(dst_dir):
        print(f"Error: destination directory already exists → {dst_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        # perform a recursive copy of the directory
        shutil.copytree(src_dir, dst_dir)
        print(f"Copy succeeded: {src_dir} → {dst_dir}")
    except Exception as e:
        print(f"Copy failed: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Copy 'outputs' to 'outputs_renamed' within the given root directory."
    )
    parser.add_argument(
        "root_dir",
        help="Path to the root directory containing 'outputs'"
    )
    args = parser.parse_args()

    copy_and_rename_outputs(args.root_dir)

if __name__ == "__main__":
    main()
