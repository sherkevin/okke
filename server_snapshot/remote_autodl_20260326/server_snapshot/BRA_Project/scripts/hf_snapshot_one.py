#!/usr/bin/env python3
"""单仓库 snapshot_download，支持断点续传；日志请由外层 tee 接管。"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    from huggingface_hub import snapshot_download

    print(f"[hf_snapshot_one] START repo={args.repo} dest={args.dest} type={args.repo_type}", flush=True)
    # local_dir 写入时默认断点续传（已存在文件会校验并跳过完整分片）
    snapshot_download(
        repo_id=args.repo,
        repo_type=args.repo_type,
        local_dir=args.dest,
        max_workers=args.max_workers,
    )
    print(f"[hf_snapshot_one] DONE repo={args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[hf_snapshot_one] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as e:
        print(f"[hf_snapshot_one] FAIL: {e!r}", file=sys.stderr, flush=True)
        raise SystemExit(1)
