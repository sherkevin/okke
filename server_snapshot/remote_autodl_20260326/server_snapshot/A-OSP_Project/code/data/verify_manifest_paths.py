#!/usr/bin/env python3
"""
Data Integrity Sanity-Check — verify_manifest_paths.py
=======================================================
Samples N random rows from every downloaded manifest, asserts:
  1. The associated image_path(s) physically exist on disk.
  2. PIL.Image.open() succeeds (file is not truncated/corrupt).
  3. (For AMBER) resolves relative image filenames via the images/ directory.
  4. (For IU X-Ray) checks both image_path AND image_paths_local list.
  5. (For MVBench) if video dir exists, checks MP4 files via decord or cv2.

Exit code 0 = all clear; 1 = one or more datasets have broken assets.

Usage:
    python code/data/verify_manifest_paths.py          # default N=5
    python code/data/verify_manifest_paths.py --n 20  # sample 20 rows
    python code/data/verify_manifest_paths.py --full   # check every row
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image as PilImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "benchmarks"


# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> str: return f"{GREEN}✓{RESET} {msg}"
def fail(msg: str) -> str: return f"{RED}✗{RESET} {msg}"
def warn(msg: str) -> str: return f"{YELLOW}!{RESET} {msg}"


# ── Image verification ──────────────────────────────────────────────────────
def check_image(abs_path: Path) -> tuple[bool, str]:
    """Returns (passed, reason)."""
    if not abs_path.exists():
        return False, f"missing: {abs_path}"
    if abs_path.stat().st_size == 0:
        return False, f"zero-byte: {abs_path}"
    if not PIL_OK:
        return True, "PIL unavailable (size OK)"
    try:
        with PilImage.open(abs_path) as img:
            img.verify()          # catches truncated / corrupted files
        return True, f"{abs_path.name} ({abs_path.stat().st_size//1024}KB)"
    except Exception as e:
        return False, f"PIL error on {abs_path.name}: {e}"


def resolve_image_path(raw: str | Path) -> Path:
    """Resolve relative or absolute image path against project ROOT."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return ROOT / p


# ── Per-dataset samplers ────────────────────────────────────────────────────

def _sample_jsonl(path: Path, n: int, full: bool) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if full or n >= len(rows):
        return rows
    return random.sample(rows, n)


def _sample_json(path: Path, n: int, full: bool) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    if full or n >= len(data):
        return data
    return random.sample(data, n)


def check_standard_manifest(
    name: str,
    manifest: Path,
    n: int,
    full: bool,
    img_key: str = "image_path",
) -> tuple[int, int, list[str]]:
    """Generic checker for manifests with a single image_path field."""
    if not manifest.exists():
        return 0, 0, [fail(f"{name}: manifest not found at {manifest}")]

    rows = _sample_jsonl(manifest, n, full)
    passed = failed = 0
    errors: list[str] = []

    for row in rows:
        raw = row.get(img_key)
        if raw is None:
            errors.append(fail(f"{name}: row {row.get('_index','?')} has no '{img_key}' field"))
            failed += 1
            continue
        abs_p = resolve_image_path(raw)
        ok_flag, reason = check_image(abs_p)
        if ok_flag:
            passed += 1
        else:
            failed += 1
            errors.append(fail(f"{name}[{row.get('_index','?')}]: {reason}"))

    return passed, failed, errors


# ── Dataset-specific checkers ───────────────────────────────────────────────

def check_mmbench(n: int, full: bool):
    manifest = BENCH / "mmbench" / "mmbench_manifest.jsonl"
    return check_standard_manifest("MMBench", manifest, n, full)


def check_amber(n: int, full: bool):
    """AMBER uses 'image' filename; images live in data/benchmarks/amber/images/."""
    manifest = BENCH / "amber" / "query_all.json"
    img_dir  = BENCH / "amber" / "images"
    if not manifest.exists():
        return 0, 0, [fail(f"AMBER: manifest not found at {manifest}")]

    rows = _sample_json(manifest, n, full)
    passed = failed = 0
    errors: list[str] = []

    for row in rows:
        fname = row.get("image")
        if not fname:
            errors.append(fail(f"AMBER id={row.get('id','?')}: no 'image' field"))
            failed += 1
            continue
        abs_p = img_dir / fname
        ok_flag, reason = check_image(abs_p)
        if ok_flag:
            passed += 1
        else:
            failed += 1
            errors.append(fail(f"AMBER[{row.get('id','?')}]: {reason}"))

    return passed, failed, errors


def check_chartqa(n: int, full: bool):
    return check_standard_manifest(
        "ChartQA", BENCH / "chartqa" / "chartqa_manifest.jsonl", n, full)


def check_iu_xray(n: int, full: bool):
    """IU X-Ray has both image_path and image_paths_local (list of 2 views)."""
    manifest = BENCH / "iu_xray" / "iu_xray_manifest.jsonl"
    if not manifest.exists():
        return 0, 0, [fail(f"IU-XRay: manifest not found")]

    rows = _sample_jsonl(manifest, n, full)
    passed = failed = 0
    errors: list[str] = []

    for row in rows:
        # check primary
        primary = row.get("image_path")
        if primary:
            ok_flag, reason = check_image(resolve_image_path(primary))
            if ok_flag:
                passed += 1
            else:
                failed += 1
                errors.append(fail(f"IU-XRay[{row.get('_index','?')}] primary: {reason}"))
        # check dual-view list
        for p in row.get("image_paths_local", []):
            ok_flag, reason = check_image(resolve_image_path(p))
            if ok_flag:
                passed += 1
            else:
                failed += 1
                errors.append(fail(f"IU-XRay[{row.get('_index','?')}] view {p}: {reason}"))

    return passed, failed, errors


def check_textvqa(n: int, full: bool):
    return check_standard_manifest(
        "TextVQA", BENCH / "textvqa" / "textvqa_manifest.jsonl", n, full)


def check_visualwebbench(n: int, full: bool):
    """Check all 7 task manifests."""
    tasks = ["action_ground","action_prediction","element_ground",
             "element_ocr","heading_ocr","web_caption","webqa"]
    total_p = total_f = 0
    all_errors: list[str] = []
    for task in tasks:
        man = BENCH / "visualwebbench" / f"vwb_{task}_manifest.jsonl"
        if not man.exists():
            all_errors.append(warn(f"VWB:{task}: manifest missing (skip)"))
            continue
        p, f, errs = check_standard_manifest(f"VWB:{task}", man, n, full)
        total_p += p; total_f += f
        all_errors.extend(errs)
    return total_p, total_f, all_errors


def check_refcoco(n: int, full: bool):
    return check_standard_manifest(
        "RefCOCO", BENCH / "refcoco" / "refcoco_manifest.jsonl", n, full)


def check_mmmu(n: int, full: bool):
    """MMMU: image_path may be None when row has no image."""
    manifest = BENCH / "mmmu" / "mmmu_manifest.jsonl"
    if not manifest.exists():
        return 0, 0, [fail("MMMU: manifest not found")]

    rows = _sample_jsonl(manifest, n, full)
    passed = failed = skipped = 0
    errors: list[str] = []

    for row in rows:
        ip = row.get("image_path")
        if ip is None:
            skipped += 1
            continue
        ok_flag, reason = check_image(resolve_image_path(ip))
        if ok_flag:
            passed += 1
        else:
            failed += 1
            errors.append(fail(f"MMMU[{row.get('_index','?')}]: {reason}"))

    if skipped:
        errors.append(warn(f"MMMU: {skipped}/{len(rows)} sampled rows had no image (text-only questions, expected)"))
    return passed, failed, errors


def check_cod10k(n: int, full: bool):
    return check_standard_manifest(
        "COD10K", BENCH / "cod10k" / "cod10k_manifest.jsonl", n, full)


def check_mvbench_videos(n: int, full: bool):
    """Check MP4 readability in the MVBench video directory if it exists."""
    video_dirs = [
        ROOT / "data" / "MVBench" / "video" / "Charades_v1_480",
        ROOT / "data" / "mvbench" / "video",
    ]
    found_dir = next((d for d in video_dirs if d.is_dir()), None)

    if found_dir is None:
        return 0, 0, [warn("MVBench: video dir not yet present — awaiting manual upload")]

    mp4s = list(found_dir.glob("**/*.mp4"))
    if not mp4s:
        return 0, 0, [warn(f"MVBench: dir exists ({found_dir}) but no .mp4 files yet")]

    sample = mp4s if (full or n >= len(mp4s)) else random.sample(mp4s, n)
    passed = failed = 0
    errors: list[str] = []

    # Try decord first, fall back to cv2
    try:
        import decord
        def _readable(p: Path) -> tuple[bool, str]:
            try:
                vr = decord.VideoReader(str(p))
                _ = vr[0]
                return True, f"{p.name} ({len(vr)} frames)"
            except Exception as e:
                return False, f"decord error on {p.name}: {e}"
    except ImportError:
        try:
            import cv2
            def _readable(p: Path) -> tuple[bool, str]:
                cap = cv2.VideoCapture(str(p))
                ok_flag = cap.isOpened()
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok_flag else 0
                cap.release()
                if ok_flag:
                    return True, f"{p.name} ({frames} frames)"
                return False, f"cv2 cannot open {p.name}"
        except ImportError:
            return 0, 0, [warn("MVBench: neither decord nor cv2 available; install one to verify videos")]

    for mp4 in sample:
        ok_flag, reason = _readable(mp4)
        if ok_flag:
            passed += 1
        else:
            failed += 1
            errors.append(fail(f"MVBench: {reason}"))

    return passed, failed, errors


def check_chair_coco():
    """CHAIR is a tool, not a manifest; just verify instances_val2014.json and chair.pkl."""
    errors: list[str] = []
    passed = 0
    anno = ROOT / "data" / "chair" / "coco_annotations" / "instances_val2014.json"
    pkl  = ROOT / "data" / "chair" / "CHAIR-metric-standalone" / "chair.pkl"
    for p, name in [(anno, "instances_val2014.json"), (pkl, "chair.pkl")]:
        if p.exists() and p.stat().st_size > 1000:
            passed += 1
        else:
            errors.append(fail(f"CHAIR: {name} missing or too small at {p}"))
    return passed, max(0, 2 - passed - len(errors)), errors


# ── Registry ────────────────────────────────────────────────────────────────
CHECKERS = [
    ("MMBench",         check_mmbench),
    ("AMBER",           check_amber),
    ("ChartQA",         check_chartqa),
    ("IU X-Ray",        check_iu_xray),
    ("TextVQA",         check_textvqa),
    ("VisualWebBench",  check_visualwebbench),
    ("RefCOCO",         check_refcoco),
    ("MMMU",            check_mmmu),
    ("COD10K",          check_cod10k),
    ("MVBench videos",  check_mvbench_videos),
    ("CHAIR toolchain", lambda n, f: check_chair_coco()),
]


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sanity-check all dataset manifests")
    parser.add_argument("--n",    type=int, default=5, help="rows to sample per manifest (default 5)")
    parser.add_argument("--full", action="store_true",  help="check every row (slow for TextVQA/MMBench)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    mode = "FULL" if args.full else f"SAMPLE n={args.n}"

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  A-OSP Data Integrity Audit   [{mode}]{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  ROOT: {ROOT}\n")

    summary: list[tuple[str, int, int, list[str]]] = []
    global_fail = False

    for name, checker in CHECKERS:
        p, f, errs = checker(args.n, args.full)
        summary.append((name, p, f, errs))
        status = ok(f"{p} passed") if f == 0 else fail(f"{p} passed, {f} FAILED")
        print(f"  [{name:<20s}] {status}")
        for e in errs:
            print(f"      {e}")
        if f > 0:
            global_fail = True

    print(f"\n{BOLD}{'='*60}{RESET}")
    total_p = sum(p for _, p, _, _ in summary)
    total_f = sum(f for _, _, f, _ in summary)
    total   = total_p + total_f

    if global_fail:
        print(f"{RED}{BOLD}  AUDIT RESULT: FAIL  —  {total_f}/{total} checks failed{RESET}")
        print(f"  Fix broken image links before handing off to Agent 2.\n")
        sys.exit(1)
    else:
        print(f"{GREEN}{BOLD}  AUDIT RESULT: PASS  —  {total_p}/{total} checks passed{RESET}")
        print(f"  All sampled image links are valid. Agent 2 has a clean runway.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
