#!/usr/bin/env python3
"""
MVBench Video Upload Monitor — monitor_mvbench_upload.py
=========================================================
Polls the Charades_v1_480 directory every INTERVAL seconds.
Once MP4 files appear (Lead Author manual upload), it:
  1. Counts and lists all .mp4 files.
  2. Samples up to SAMPLE files and verifies readability via decord (fallback cv2).
  3. Prints a pass/fail report and exits with code 0 on success.

Usage:
    # Background watch (polls every 60s, exits on success):
    python code/data/monitor_mvbench_upload.py --interval 60 &

    # One-shot check (exit immediately with current status):
    python code/data/monitor_mvbench_upload.py --once

    # Verify a specific dir:
    python code/data/monitor_mvbench_upload.py --dir /path/to/videos --once
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DIRS = [
    ROOT / "data/MVBench/video/Charades_v1_480",
    ROOT / "data/MVBench/video",
    ROOT / "data/mvbench/video",
]


def _get_reader():
    """Return a callable(path)->( bool, str ) using decord or cv2."""
    try:
        import decord
        def read(p: Path):
            try:
                vr = decord.VideoReader(str(p))
                _ = vr[0]
                return True, f"{p.name}: {len(vr)} frames, OK (decord)"
            except Exception as e:
                return False, f"{p.name}: decord error — {e}"
        return read
    except ImportError:
        pass
    try:
        import cv2
        def read(p: Path):
            cap = cv2.VideoCapture(str(p))
            ok = cap.isOpened()
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
            cap.release()
            if ok:
                return True, f"{p.name}: {frames} frames, OK (cv2)"
            return False, f"{p.name}: cv2 cannot open"
        return read
    except ImportError:
        return None


def check_once(video_dir: Path, sample: int) -> tuple[bool, str]:
    if not video_dir.is_dir():
        return False, f"dir not found: {video_dir}"
    mp4s = sorted(video_dir.glob("**/*.mp4"))
    if not mp4s:
        return False, f"dir exists but no .mp4 files yet: {video_dir}"

    reader = _get_reader()
    if reader is None:
        return False, "neither decord nor cv2 installed; run: pip install decord"

    import random
    sample_files = mp4s[:sample] if sample >= len(mp4s) else random.sample(mp4s, sample)
    passed = failed = 0
    lines = [f"Found {len(mp4s)} .mp4 files in {video_dir}"]
    for p in sample_files:
        ok, msg = reader(p)
        if ok:
            passed += 1
            lines.append(f"  ✓ {msg}")
        else:
            failed += 1
            lines.append(f"  ✗ {msg}")

    if failed == 0:
        lines.append(f"\nSAMPLE CHECK: {passed}/{len(sample_files)} passed — VIDEO STORE READY ✅")
        return True, "\n".join(lines)
    else:
        lines.append(f"\nSAMPLE CHECK: {failed}/{len(sample_files)} FAILED")
        return False, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",      default=None, help="override video directory")
    parser.add_argument("--interval", type=int, default=60, help="poll interval seconds (default 60)")
    parser.add_argument("--sample",   type=int, default=10, help="MP4 files to verify (default 10)")
    parser.add_argument("--once",     action="store_true", help="check once and exit")
    args = parser.parse_args()

    if args.dir:
        dirs = [Path(args.dir)]
    else:
        dirs = DEFAULT_DIRS

    if args.once:
        for d in dirs:
            ok, msg = check_once(d, args.sample)
            print(msg)
            if ok:
                sys.exit(0)
        sys.exit(1)

    # Polling loop
    print(f"[mvbench-monitor] Watching {[str(d) for d in dirs]}")
    print(f"[mvbench-monitor] Poll interval: {args.interval}s | Sample: {args.sample} files")
    while True:
        for d in dirs:
            ok, msg = check_once(d, args.sample)
            ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")
            if ok:
                print(f"[mvbench-monitor] Upload complete — exiting.")
                sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
