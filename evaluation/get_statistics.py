#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import json

def analyze_log(log_file):
    """
    Returns:
        is_passed (bool): True if log contains no 'Failed' and at least one 'Passed'
        fail_count (int): number of 'Failed' occurrences
    """
    try:
        content = open(log_file, 'r', encoding='utf-8').read()
        fail_count = len(re.findall(r'Failed', content))
        pass_count = len(re.findall(r'Passed', content))
    except Exception:
        # on any error, consider it a failure with one failure count
        return False, 1

    # only count as passed if there are no 'Failed' AND at least one 'Passed'
    is_passed = (fail_count == 0 and pass_count > 0)
    return is_passed, fail_count


def main():
    parser = argparse.ArgumentParser(description="统计 unit_test 的通过/失败情况（只看 FAIL）")
    parser.add_argument('--eval_dir',  required=True, help="测试结果输出目录，各子目录的repo_name下应有 unit_test/logs/*.log")
    parser.add_argument('--output_dir', required=True, help="输出目录，results.txt 将写入此目录")
    args = parser.parse_args()

    eval_dir   = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'results.txt'

    total_files = 0
    passed_files = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for sub in sorted(eval_dir.iterdir()):
            info_file = sub / 'info.json'
            if not info_file.is_file():
                print(f"[WARN] skip {sub.name} due to lack of info.json")
                continue
            # 提取子目录名，用于前缀文件名
            folder_name = sub.name

            info = json.loads(info_file.read_text(encoding='utf-8'))
            repo_name = info.get('repo_folder_name')
            repo_res = eval_dir / sub.name / repo_name / 'unit_test' / 'logs'
            if not repo_res.is_dir():
                continue

            for fname in sorted(os.listdir(repo_res)):
                if not fname.endswith('.log'):
                    continue
                total_files += 1
                fpath = repo_res / fname

                is_passed, fail_count = analyze_log(fpath)
                # 在文件名前加入子目录名
                if is_passed:
                    out.write(f"{folder_name}/{fname}: PASS\n")
                    passed_files += 1
                else:
                    out.write(f"{folder_name}/{fname}: FAIL (found {fail_count} failure lines)\n")

        # 汇总
        out.write("\n")
        out.write(f"Total log files: {total_files}\n")
        out.write(f"Passed files:    {passed_files}\n")
        out.write(f"Failed files:    {total_files - passed_files}\n")
        if total_files > 0:
            rate = passed_files / total_files * 100
            out.write(f"Overall pass rate: {rate:.2f}%\n")

if __name__ == '__main__':
    main()