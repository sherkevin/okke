#!/usr/bin/env python3
"""单个子任务入口，供并行下载编排器调用（避免重复加载整份 pipeline）。"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_MOD_PATH = Path(__file__).parent / "download_experiment_datasets.py"


def _load_ded():
    spec = importlib.util.spec_from_file_location("download_experiment_datasets", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "task",
        choices=("vwb", "refcoco", "mmmu", "mvbench", "mirage", "cod10k"),
    )
    p.add_argument("--n_vwb", type=int, default=500)
    p.add_argument("--n_refcoco", type=int, default=200)
    p.add_argument("--n_mmmu", type=int, default=500)
    p.add_argument("--n_mvbench", type=int, default=200)
    args = p.parse_args()

    m = _load_ded()
    ok = True
    if args.task == "vwb":
        ok = m.download_visualwebbench_mini(args.n_vwb, m.BENCH / "visualwebbench")
    elif args.task == "refcoco":
        ok = m.download_refcoco_mini(args.n_refcoco, m.BENCH / "refcoco")
    elif args.task == "mmmu":
        ok = m.download_mmmu_mini(args.n_mmmu, m.BENCH / "mmmu")
    elif args.task == "mvbench":
        ok = m.download_mvbench_manifests(args.n_mvbench, m.DATA / "mvbench")
    elif args.task == "mirage":
        ok = m.ensure_mirage_stub()
    elif args.task == "cod10k":
        m.write_cod10k_readme()
        ok = True

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
