#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from launch_pope_chair_main_table import SSH_HOST, SSH_KEY, SSH_PORT, run_remote_bash


def run_local(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def remote_exists(remote_path: str) -> bool:
    script = f"""set -euo pipefail
if [ -e "{remote_path}" ]; then
  echo EXISTS
else
  echo MISSING
fi
"""
    proc = run_remote_bash(script, timeout=120, check=False)
    return "EXISTS" in proc.stdout


def ensure_remote_parent(remote_path: str) -> None:
    parent = str(Path(remote_path).parent).replace("\\", "/")
    script = f"""set -euo pipefail
mkdir -p "{parent}"
"""
    proc = run_remote_bash(script, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create remote parent for {remote_path}:\n{proc.stdout}\n{proc.stderr}")


def scp_recursive(local_path: Path, remote_path: str) -> None:
    cmd = [
        "scp",
        "-r",
        "-i",
        SSH_KEY,
        "-P",
        SSH_PORT,
        "-o",
        "StrictHostKeyChecking=no",
        str(local_path),
        f"{SSH_HOST}:{remote_path}",
    ]
    run_local(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model locally, upload to remote, then optionally delete local staging.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-path", required=True, help="Full remote destination directory path.")
    parser.add_argument("--staging-root", default=r"D:\hf_model_staging")
    parser.add_argument("--delete-local-after-upload", action="store_true")
    parser.add_argument("--delete-cache-after-upload", action="store_true")
    parser.add_argument("--skip-if-remote-exists", action="store_true")
    args = parser.parse_args()

    remote_path = args.remote_path.replace("\\", "/")
    if args.skip_if_remote_exists and remote_exists(remote_path):
        print(f"Remote path already exists, skipping download/upload: {remote_path}")
        return 0

    staging_root = Path(args.staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    target_name = Path(remote_path).name
    cache_dir = staging_root / f".hf_cache_{target_name}"
    local_dir = staging_root / target_name

    print(f"Downloading {args.repo_id} to {local_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        resume_download=True,
        local_dir_use_symlinks=False,
    )

    ensure_remote_parent(remote_path)
    print(f"Uploading {local_dir} -> {remote_path} ...")
    scp_recursive(local_dir, remote_path)

    if args.delete_local_after_upload and local_dir.exists():
        print(f"Deleting local staging dir {local_dir} ...")
        shutil.rmtree(local_dir)

    if args.delete_cache_after_upload and cache_dir.exists():
        print(f"Deleting local cache dir {cache_dir} ...")
        shutil.rmtree(cache_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
