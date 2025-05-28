#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

def image_exists_on_dockerhub(image_tag: str) -> bool:
    """
    Return True if `docker manifest inspect <image_tag>` succeeds,
    meaning the image:tag already exists on Docker Hub.
    """
    result = subprocess.run(
        ['docker', 'manifest', 'inspect', image_tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def main(root_dir: str):
    # Get the current user’s UID and GID
    user_id = os.getuid()
    group_id = os.getgid()

    # Read Docker Hub username from environment
    dockerhub_username = os.environ.get('DOCKERHUB_USERNAME')
    if not dockerhub_username:
        raise RuntimeError(
            "Please set DOCKERHUB_USERNAME in your environment, e.g.:\n"
            "export DOCKERHUB_USERNAME=your_username"
        )

    root = Path(root_dir)
    if not root.is_dir():
        raise RuntimeError(f"Root directory does not exist or is not a directory: {root_dir}")

    for subfolder in tqdm(root.iterdir()):
        if not subfolder.is_dir():
            continue

        info_path = subfolder / 'info.json'
        dockerfile_path = subfolder / 'Dockerfile'
        if not info_path.exists() or not dockerfile_path.exists():
            print(f"[SKIP] {subfolder.name}: missing info.json or Dockerfile")
            continue

        # Load info.json
        info = json.loads(info_path.read_text(encoding='utf-8'))
        folder_name = info.get('folder_name')
        repo_folder_name = info.get('repo_folder_name')
        if not folder_name or not repo_folder_name:
            print(f"[SKIP] {subfolder.name}: 'folder_name' or 'repo_folder_name' not found in info.json")
            continue

        repo_path = subfolder / repo_folder_name
        if not repo_path.exists():
            print(f"[WARN] {subfolder.name}: repo folder '{repo_folder_name}' not found")

        # sanitize: lowercase (Docker Hub requires lowercase repos)
        folder_name = folder_name.lower()
        image_tag = f"{dockerhub_username}/{folder_name}:latest"

        # skip if already on Docker Hub
        print(f"[CHECK] {image_tag}")
        if image_exists_on_dockerhub(image_tag):
            print(f"[SKIP] {image_tag} already exists on Docker Hub")
            continue


        print(f"[BUILD] Building image {image_tag} in context {subfolder}")
        subprocess.run([
            'docker', 'build',
            '--build-arg', f"UID={user_id}",
            '--build-arg', f"GID={group_id}",
            '--build-arg', f"DIR={repo_folder_name}",
            '-t', image_tag,
            '-f', str(dockerfile_path),
            '.'
        ], cwd=str(subfolder), check=True)

        print(f"[PUSH] Pushing image {image_tag} to Docker Hub")
        subprocess.run(['docker', 'push', image_tag], check=True)

    print("All done.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-build and push Docker images: each subdirectory must contain info.json, Dockerfile, and the specified repo folder."
    )
    parser.add_argument(
        'root_dir',
        help="Path to the root directory containing multiple project subfolders"
    )
    args = parser.parse_args()
    main(args.root_dir)
