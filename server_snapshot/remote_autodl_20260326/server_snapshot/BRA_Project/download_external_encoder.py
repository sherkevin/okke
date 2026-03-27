#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a frozen external encoder snapshot.")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. openai/clip-vit-large-patch14")
    parser.add_argument("--output", required=True, help="Local output directory")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF mirror endpoint")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache dir")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(path)


if __name__ == "__main__":
    main()
