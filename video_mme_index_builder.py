#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a robust Video-MME video index.")
    parser.add_argument("--root", default="/root/autodl-tmp/BRA_Project/datasets/video/Video-MME_hf")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "video_mme_index.json"
    cache_dir = Path(args.cache_dir) if args.cache_dir else root / ".bra_cache" / "video_mme"

    entries = {}
    actual_files = 0
    zip_members = 0

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            actual_files += 1
            entries[path.stem] = {
                "video_id": path.stem,
                "resolved_video_path": str(path),
                "source_chunk_or_zip": str(path),
            }

    for zip_path in sorted(root.glob("videos_chunked_*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    member_path = Path(member)
                    if member_path.suffix.lower() not in VIDEO_SUFFIXES:
                        continue
                    zip_members += 1
                    video_id = member_path.stem
                    if video_id in entries:
                        continue
                    entries[video_id] = {
                        "video_id": video_id,
                        "resolved_video_path": str(cache_dir / member_path.name),
                        "source_chunk_or_zip": str(zip_path),
                        "zip_member": member,
                    }
        except zipfile.BadZipFile:
            continue

    payload = {
        "root": str(root),
        "cache_dir": str(cache_dir),
        "stats": {
            "video_count": len(entries),
            "actual_files": actual_files,
            "zip_members": zip_members,
        },
        "entries": [entries[key] for key in sorted(entries)],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved Video-MME index to {output}")
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
