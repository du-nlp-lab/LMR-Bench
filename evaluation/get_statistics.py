#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import json

def analyze_log(log_file):
    """
    Analyze a single test log file.

    Returns:
        is_passed (bool): True if there are no 'Failed' entries and at least one 'Passed' entry.
        fail_count (int): Number of 'Failed' occurrences.
    """
    try:
        content = open(log_file, 'r', encoding='utf-8').read()
        fail_count = len(re.findall(r'Failed', content))
        pass_count = len(re.findall(r'Passed', content))
    except Exception:
        # If any error occurs reading or parsing the file, treat it as a failed test
        return False, 1

    # Only consider the log as passed if there are zero failures and at least one pass
    is_passed = (fail_count == 0 and pass_count > 0)
    return is_passed, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Summarize unit test pass/fail results (only 'FAIL' lines are counted)."
    )
    parser.add_argument(
        '--eval_dir', required=True,
        help="Root directory of test outputs. Each subdirectory should contain info.json and a 'unit_test/logs' folder."
    )
    parser.add_argument(
        '--output_dir', required=True,
        help="Directory where 'results.txt' will be created."
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / 'results.txt'

    total_files = 0
    passed_files = 0

    with open(results_path, 'w', encoding='utf-8') as out:
        for case_dir in sorted(eval_dir.iterdir()):
            info_path = case_dir / 'info.json'
            if not info_path.is_file():
                out.write(f"[WARNING] Skipping '{case_dir.name}': info.json not found\n")
                # total_files += 1
                continue

            # Load repository folder name from metadata
            metadata = json.loads(info_path.read_text(encoding='utf-8'))
            repo_folder = metadata.get('repo_folder_name')
            logs_dir = case_dir / repo_folder / 'unit_test' / 'logs'

            if not logs_dir.is_dir():
                out.write(f"{logs_dir} is not a directory\n")
                total_files += 1
                continue

            for fname in sorted(os.listdir(logs_dir)):
                if not fname.endswith('.log'):
                    out.write(f"{fname} is not a log file\n")
                    continue

                total_files += 1
                log_path = logs_dir / fname
                is_passed, fail_count = analyze_log(log_path)
                prefix = case_dir.name

                if is_passed:
                    out.write(f"{prefix}/{fname}: PASS (fail_count={fail_count})\n")
                    passed_files += 1
                else:
                    out.write(f"{prefix}/{fname}: FAIL (found {fail_count} failure lines)\n")

        # Write summary
        out.write("\n")
        out.write(f"Total log files: {total_files}\n")
        out.write(f"Passed files:    {passed_files}\n")
        out.write(f"Failed files:    {total_files - passed_files}\n")
        if total_files > 0:
            pass_rate = passed_files / total_files * 100
            out.write(f"Overall pass rate: {pass_rate:.2f}%\n")


if __name__ == '__main__':
    main()
