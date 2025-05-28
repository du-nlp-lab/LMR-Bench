#!/usr/bin/env python3
import argparse
import sys
import json
import subprocess
from pathlib import Path
import os

user_id = os.getuid()
group_id = os.getgid()


def try_pull(tag: str) -> bool:
    return subprocess.run(
        ["docker", "pull", tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def main():
    parser = argparse.ArgumentParser(description="批量运行 Docker 容器并执行单元测试")
    parser.add_argument('--evaluation_dir', required=True, help="评估文件夹的根路径")
    parser.add_argument('--evaluation_output_dir', required=True, help="测试结果输出到此路径")
    args = parser.parse_args()

    eval_root = Path(args.evaluation_dir)
    output_root = Path(args.evaluation_output_dir)
    user="shinyy1997"

    # 1. 检查评估目录
    if not eval_root.is_dir():
        sys.exit(f"Error: No directory: {eval_root}")

    # 2. 确保输出目录存在
    output_root.mkdir(parents=True, exist_ok=True)

    # 3. 遍历子文件夹
    for subfolder in sorted(eval_root.iterdir()):
        if not subfolder.is_dir():
            continue


        sub_out = output_root / subfolder.name
        # 如果存在并且非空，则认为已经评估过，跳过
        if sub_out.exists() and any(sub_out.iterdir()):
            print(f"[INFO] skip {subfolder.name} (already evaluated)")
            continue

        info_file = subfolder / 'info.json'
        if not info_file.is_file():
            print(f"[WARN] skip {subfolder.name} due to lack of info.json")
            continue

        info = json.loads(info_file.read_text(encoding='utf-8'))
        repo_name = info.get('repo_folder_name')

        dockerfile = subfolder / 'Dockerfile'
        image_tag = f"eval_{subfolder.name.lower()}"

        # hub_repo   = "yourhubusername/benchmark"
        # image_tag  = f"{hub_repo}:{subfolder.name.lower()}"
        image_tag = f"{user}/{subfolder.name.lower()}:latest"

        print(f"[INFO] try to pull docker image {image_tag} from Docker Hub …")
        if try_pull(image_tag):
            print(f"[INFO] pulled existing image {image_tag}")
        else:
            print(f"[INFO] pull failed, building docker image {image_tag} …")
            subprocess.run([
                'docker', 'build',
                '--build-arg', f"UID={user_id}",
                '--build-arg', f"GID={group_id}",
                '--build-arg', f"DIR={repo_name}",
                '-t', image_tag,
                '-f', str(dockerfile),
                '.'
            ], cwd=subfolder, check=True)

        # 5. Docker run
        print(f"[INFO] run the container {image_tag}")
        docker_cmd = [
            'docker', 'run', '--rm',
            '--user', f'{user_id}:{group_id}',
            '--gpus', 'device=0',
            '-e', f"HUGGINGFACE_HUB_TOKEN={os.environ['HUGGINGFACE_HUB_TOKEN']}", 
            '-e', f"HF_TOKEN={os.environ['HUGGINGFACE_HUB_TOKEN']}", 
            '-v', f"{subfolder.resolve()}:/workspace",
            '-v', "/home/sxy240002/tmp:/tmp",
            '-v', "/home/sxy240002/transformers_cache:/home/user/.cache",
            image_tag,
            'bash', '-lc',
            (
                f"pwd && cd {repo_name} && "
                "export PYTHONPATH=$(pwd) && "
                "mkdir -p /workspace/results && "  
                "for pyf in unit_test/unit_test_*.py; do "
                "python \"$pyf\" > /workspace/results/\"$(basename \"$pyf\" .py)\".log 2>&1; "
                "done"
            )
        ]

        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] run failed for {subfolder.name}: {e}")

            # 1. Copy the entire project folder to output_root/<subfolder.name>
            dest_folder = output_root / subfolder.name
            import shutil
            shutil.copytree(str(subfolder), str(dest_folder))

            # 2. Under the copied repo_name/unit_test/logs, create failure logs
            repo_dir = dest_folder / repo_name
            logs_dir = repo_dir / 'unit_test' / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Enumerate all unit_test_*.py files and generate a "Test Failed" log for each
            test_dir = repo_dir / 'unit_test'
            test_files = sorted(test_dir.glob('unit_test_*.py'))
            for idx, _ in enumerate(test_files, start=1):
                log_file = logs_dir / f"unit_test_{idx}.log"
                if not log_file.exists() or log_file.stat().st_size == 0:
                    log_file.write_text("Test Failed", encoding='utf-8')

            # Skip to the next project
            continue


        # 运行完毕后，将容器内的 results 目录复制到宿主机 output_root
        local_result_dir = output_root / subfolder.name / 'results'
        local_result_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'cp', '-r',
            str(subfolder / 'results'),
            str(local_result_dir)
        ], check=True)

    # 6. 统计结果
    print("[INFO] 执行统计脚本 get_statistics.py ...")
    subprocess.run([
        'python3', 'evaluation/get_statistics.py',
        '--eval_dir', str(eval_root), "--output_dir", str(output_root)
    ], check=True)

if __name__ == '__main__':
    main()
