#!/usr/bin/env python3
import argparse
import sys
import json
import subprocess
from pathlib import Path
import os
import shutil

user_id = os.getuid()
group_id = os.getgid()


def try_pull(tag: str) -> bool:
    """
    Try to pull a Docker image by tag.
    Returns True if the pull succeeds, False otherwise.
    """
    return subprocess.run(
        ["docker", "pull", tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run Docker containers and execute unit tests"
    )
    parser.add_argument(
        '--evaluation_dir',
        required=True,
        help="Root path of the evaluation directory"
    )
    parser.add_argument(
        '--evaluation_output_dir',
        required=True,
        help="Directory where test results will be saved"
    )
    args = parser.parse_args()

    eval_root = Path(args.evaluation_dir)
    output_root = Path(args.evaluation_output_dir)
    docker_user = "shinyy1997"

    # 1. Verify that the evaluation directory exists
    if not eval_root.is_dir():
        sys.exit(f"Error: Evaluation directory not found: {eval_root}")

    # 2. Ensure the output directory exists
    output_root.mkdir(parents=True, exist_ok=True)

    # 3. Iterate over each subdirectory in the evaluation root
    for subfolder in sorted(eval_root.iterdir()):
        if not subfolder.is_dir():
            continue

        sub_out = output_root / subfolder.name
        # If already evaluated (non-empty), skip
        if sub_out.exists() and any(sub_out.iterdir()):
            print(f"[INFO] Skipping {subfolder.name} (already evaluated)")
            continue

        info_file = subfolder / 'info.json'
        if not info_file.is_file():
            print(f"[WARN] Skipping {subfolder.name}: info.json not found")
            continue

        info = json.loads(info_file.read_text(encoding='utf-8'))
        repo_name = info.get('repo_folder_name')

        dockerfile = subfolder / 'Dockerfile'
        image_tag = f"{docker_user}/{subfolder.name.lower()}:latest"

        print(f"[INFO] Attempting to pull Docker image {image_tag} from Docker Hub…")
        if try_pull(image_tag):
            print(f"[INFO] Successfully pulled existing image {image_tag}")
        else:
            print(f"[INFO] Pull failed, building Docker image {image_tag}…")
            subprocess.run([
                'docker', 'build',
                '--build-arg', f"UID={user_id}",
                '--build-arg', f"GID={group_id}",
                '--build-arg', f"DIR={repo_name}",
                '-t', image_tag,
                '-f', str(dockerfile),
                '.'
            ], cwd=subfolder, check=True)

        # 4. Run the container and execute unit tests
        print(f"[INFO] Running container {image_tag}")
        docker_cmd = [
            'docker', 'run', '--rm',
            '--user', f'{user_id}:{group_id}',
            '--gpus', 'device=0',
            '-e', f"HUGGINGFACE_HUB_TOKEN={os.environ.get('HUGGINGFACE_HUB_TOKEN', '')}",
            '-e', f"HF_TOKEN={os.environ.get('HUGGINGFACE_HUB_TOKEN', '')}",
            '-v', f"{subfolder.resolve()}:/workspace",
            '-v', "/home/sxy240002/tmp:/tmp",
            '-v', "/home/sxy240002/transformers_cache:/home/user/.cache",
            image_tag,
            'bash', '-lc',
            (
                f"pwd && cd {repo_name} && "
                "export PYTHONPATH=$(pwd) && "
                "mkdir -p /workspace/results && "
                "for test_file in unit_test/unit_test_*.py; do "
                "python \"$test_file\" > /workspace/results/\"$(basename \"$test_file\" .py)\".log 2>&1; "
                "done"
            )
        ]
        
        run_success=True
        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            run_success = False
            print(f"[ERROR] Test run failed for {subfolder.name}: {e}")
            
            # Copy the results directory from container workspace to host output
            local_result_dir = output_root / subfolder.name / 'results'
            local_result_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                'cp', '-r',
                str(subfolder / 'results'),
                str(local_result_dir)
            ], check=True)
            error_log = local_result_dir / 'error.log'
            error_log.write_text(f"Test run failed for {subfolder.name}:\n{e}\n", encoding='utf-8')


        # Copy the entire project folder to output for inspection
        dest_folder = output_root / subfolder.name
        shutil.copytree(str(subfolder), str(dest_folder), dirs_exist_ok=True)

        # Create failure logs under <repo>/unit_test/logs
        repo_dir = dest_folder / repo_name
        logs_dir = repo_dir / 'unit_test' / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        if not run_success:
            # For each unit_test_*.py, generate a "Test Failed" log if missing or empty
            test_dir = repo_dir / 'unit_test'
            test_files = sorted(test_dir.glob('unit_test_*.py'))
            for idx, _ in enumerate(test_files, start=1):
                log_file = logs_dir / f"unit_test_{idx}.log"
                if not log_file.exists() or log_file.stat().st_size == 0:
                    log_file.write_text("Test Failed", encoding='utf-8')





    # 5. Generate summary statistics
    print("[INFO] Running the statistics script get_statistics.py…")
    subprocess.run([
        'python3', 'evaluation/get_statistics.py',
        '--eval_dir', str(eval_root),
        '--output_dir', str(output_root)
    ], check=True)


if __name__ == '__main__':
    main()
