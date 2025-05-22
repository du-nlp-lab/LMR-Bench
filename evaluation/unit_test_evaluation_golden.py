#!/usr/bin/env python3
"""
Run unit tests for a *single* main‑folder project.

Usage:
    python unit_test_evaluation.py --project_dir /path/to/main_folder
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

USER_ID = os.getuid()
GROUP_ID = os.getgid()


def build_cp_commands(repo_name: str, implementations: list[dict]) -> str:
    """Return a single Bash command that copies every golden file to its goal path."""
    pieces = []
    for impl in implementations:
        goal   = impl["goal_file"]                     # 相对 repo 路径
        golden = impl["golden_file"]                   # 相对 main‑folder 路径
        abs_golden = f'/workspace/{golden}'
        abs_goal   = f'/workspace/{repo_name}/{goal}'
        pieces.append(f'cp -f "{abs_golden}" "{abs_goal}"')
    return " && ".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Docker 中运行指定 main‑folder 的单元测试")
    parser.add_argument(
        "--project_dir",
        required=True,
        help="单个 main‑folder 的路径（包含 Dockerfile、repo 及 golden_files）",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        sys.exit(f"[ERROR] 路径不存在: {project_dir}")

    info_path = project_dir / "info.json"
    if not info_path.is_file():
        sys.exit(f"[ERROR] 缺少 info.json: {info_path}")

    info            = json.loads(info_path.read_text(encoding="utf-8"))
    repo_name       = info["repo_folder_name"]
    implementations = info.get("implementations", [])
    dockerfile      = project_dir / "Dockerfile"
    image_tag       = f"eval_{project_dir.name.lower()}"

    # ---------- build docker image ----------
    print(f"[INFO] docker build → {image_tag}")
    subprocess.run(
        [
            "docker",
            "build",
            "--build-arg",
            f"UID={USER_ID}",
            "--build-arg",
            f"GID={GROUP_ID}",
            "--build-arg",
            f"DIR={repo_name}",
            "-t",
            image_tag,
            "-f",
            str(dockerfile),
            ".",
        ],
        cwd=project_dir,
        check=True,
    )

    # ---------- prepare copy‑golden command ----------
    replace_cmd = build_cp_commands(repo_name, implementations)

    # ---------- run container and execute tests ----------
    print(f"[INFO] docker run → {image_tag}")
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{USER_ID}:{GROUP_ID}",
        "--gpus",
        "device=0",
        "-e",
        f"HUGGINGFACE_HUB_TOKEN={os.environ['HUGGINGFACE_HUB_TOKEN']}", 
        "-v",
        f"{project_dir}:/workspace",
        "-v",
        "/home/sxy240002/tmp:/tmp",
        "-v",
        "/home/sxy240002/transformers_cache:/home/user/.cache",
        image_tag,
        "bash",
        "-lc",
        (
            # ① 复制 golden → 目标文件
            f"{replace_cmd} && "
            # ② 进入 repo 并运行所有 unit_test
            f"cd {repo_name} && "
            "export PYTHONPATH=$(pwd) && "
            'for py in unit_test/unit_test_*.py; do '
            '  echo "[RUN] $py"; '
            '  python "$py"; '
            "done"
        ),
    ]
    subprocess.run(docker_cmd, check=True)

    print("[INFO] 全部 unit_test 运行完毕。日志已输出到终端。")


if __name__ == "__main__":
    main()
