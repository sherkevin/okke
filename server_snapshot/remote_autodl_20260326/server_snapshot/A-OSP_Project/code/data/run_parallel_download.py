#!/usr/bin/env python3
"""
并行拉取实验数据集（在已配置 http(s)_proxy 时走 VPN）。

默认同时启动多路子进程（互不依赖的 Hub 任务并行），缩短总墙钟时间。
日志：logs/parallel_<name>.log；汇总：logs/parallel_orchestrator.log

用法（项目根）:
  export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
  unset HF_ENDPOINT   # 或保留镜像
  python3 code/data/run_parallel_download.py

环境变量:
  SKIP_CHAIR=1          不跑 CHAIR
  SKIP_AMBER_GDOWN=1    不跑 AMBER gdown
  TEXTVQA_N=2000        TextVQA 条数
  REPORT_INTERVAL_SEC=60  进度打印间隔
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def _tee_log(path: Path):
    return open(path, "a", encoding="utf-8", buffering=1)


def _spawn(name: str, cmd: list[str], env: dict) -> subprocess.Popen:
    lp = LOG_DIR / f"parallel_{name}.log"
    f = open(lp, "w", encoding="utf-8", buffering=1)
    f.write(f"=== START {name} {datetime.now(timezone.utc).isoformat()} ===\n")
    f.write(f"CMD: {' '.join(cmd)}\n\n")
    f.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    ), lp, f


def main() -> None:
    env = os.environ.copy()
    report_iv = int(env.get("REPORT_INTERVAL_SEC", "60"))

    orch = _tee_log(LOG_DIR / "parallel_orchestrator.log")
    orch.write(f"\n{'='*60}\nORCHESTRATOR START {datetime.now(timezone.utc).isoformat()}\n")
    orch.flush()
    # 轻量依赖（并行子进程各自也会 import datasets）
    subprocess.run(
        [PY, "-m", "pip", "install", "-q", "gdown", "huggingface_hub", "hf_transfer"],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    tvqa_n = str(env.get("TEXTVQA_N", "2000"))
    jobs: list[tuple[str, list[str]]] = [
        ("core", [PY, str(ROOT / "code/data/download_core_benchmarks.py"), "--full"]),
        ("crossdomain", [PY, str(ROOT / "code/data/download_crossdomain_datasets.py"), "--full"]),
        (
            "textvqa",
            [
                PY,
                str(ROOT / "code/data/download_textvqa_stream.py"),
                tvqa_n,
                "--out",
                str(ROOT / "data/benchmarks/textvqa"),
            ],
        ),
        ("visualwebbench", [PY, str(ROOT / "code/data/run_experiment_subtask.py"), "vwb"]),
        (
            "refcoco",
            [
                PY,
                str(ROOT / "code/data/download_refcoco_stream.py"),
                env.get("REFCOCO_N", "200"),
                "--out",
                str(ROOT / "data/benchmarks/refcoco"),
            ],
        ),
        ("mmmu", [PY, str(ROOT / "code/data/run_experiment_subtask.py"), "mmmu"]),
        ("mvbench", [PY, str(ROOT / "code/data/run_experiment_subtask.py"), "mvbench"]),
        ("mirage", [PY, str(ROOT / "code/data/run_experiment_subtask.py"), "mirage"]),
        ("cod10k_readme", [PY, str(ROOT / "code/data/run_experiment_subtask.py"), "cod10k"]),
    ]

    procs: list[tuple[str, subprocess.Popen, Path, object]] = []
    for name, cmd in jobs:
        p, lp, fh = _spawn(name, cmd, env)
        procs.append((name, p, lp, fh))
        orch.write(f"spawned {name} pid={p.pid} log={lp}\n")

    chair_p = None
    chair_lp = None
    chair_fh = None
    if env.get("SKIP_CHAIR", "").lower() not in ("1", "true", "yes"):
        chair_t = env.get("CHAIR_SETUP_TIMEOUT_SEC", "7200")
        # timeout 由外层 wait 控制；这里用 timeout 包装 bash
        cmd = [
            "/usr/bin/env",
            "bash",
            "-c",
            f"timeout {chair_t} bash {ROOT}/code/data/setup_chair.sh",
        ]
        chair_p, chair_lp, chair_fh = _spawn("chair", cmd, env)
        orch.write(f"spawned chair pid={chair_p.pid} log={chair_lp}\n")

    amber_p = None
    amber_lp = None
    amber_fh = None
    if env.get("SKIP_AMBER_GDOWN", "").lower() not in ("1", "true", "yes"):
        amber_dir = ROOT / "data/benchmarks/amber/images"
        amber_dir.mkdir(parents=True, exist_ok=True)
        gurl = "https://drive.google.com/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
        cmd = [
            "/bin/bash",
            "-c",
            f"cd '{amber_dir}' && gdown '{gurl}' --fuzzy --remaining-ok",
        ]
        try:
            amber_p, amber_lp, amber_fh = _spawn("amber_gdown", cmd, env)
            orch.write(f"spawned amber_gdown pid={amber_p.pid} log={amber_lp}\n")
        except Exception as e:
            orch.write(f"amber_gdown spawn skip: {e}\n")

    all_watch = [(n, p, lp, fh) for n, p, lp, fh in procs]
    if chair_p:
        all_watch.append(("chair", chair_p, chair_lp, chair_fh))
    if amber_p:
        all_watch.append(("amber_gdown", amber_p, amber_lp, amber_fh))

    stop_evt = threading.Event()

    def reporter():
        while not stop_evt.wait(timeout=report_iv):
            orch.write(f"\n--- tick {datetime.now(timezone.utc).isoformat()} ---\n")
            for name, p, lp, _ in all_watch:
                st = "running" if p.poll() is None else f"done rc={p.poll()}"
                orch.write(f"  [{name}] {st}  log={lp.name}\n")
            orch.flush()
            # 简要磁盘
            try:
                import shutil
                du = shutil.disk_usage(ROOT / "data")
                orch.write(f"  data volume free_GB={du.free/1e9:.1f}\n")
            except OSError:
                pass
            orch.flush()

    rep_thread = threading.Thread(target=reporter, daemon=True)
    rep_thread.start()

    # 全部子进程同时跑，不按顺序等待（缩短总时间）
    while any(p.poll() is None for _, p, _, _ in all_watch):
        time.sleep(3)

    rc_map: dict[str, int] = {}
    for name, p, lp, fh in all_watch:
        rc = p.wait()
        fh.close()
        rc_map[name] = rc
        orch.write(f"FINISHED {name} rc={rc}\n")
        orch.flush()

    stop_evt.set()
    rep_thread.join(timeout=2)

    # RefCOCO：子进程常在写盘成功后因 pyarrow 析构 SIGABRT（rc=-6），manifest+jpg 完整则视为成功
    man_r = ROOT / "data/benchmarks/refcoco/refcoco_manifest.jsonl"
    if rc_map.get("refcoco") == -6 and man_r.is_file() and man_r.stat().st_size > 100:
        imgs = list((ROOT / "data/benchmarks/refcoco/images").glob("refcoco_*.jpg"))
        if len(imgs) >= 10:
            rc_map["refcoco"] = 0
            orch.write(
                "NOTE: refcoco exit code normalized to 0 (manifest + images OK; PyArrow cleanup SIGABRT)\n"
            )

    orch.write(f"\nALL_JOBS_JOINED {datetime.now(timezone.utc).isoformat()}\n")
    orch.write("RC summary (after refcoco fix if applicable): " + str(rc_map) + "\n")
    orch.write("PARALLEL_DOWNLOAD_DONE\n")
    orch.close()

    # AMBER：gdown 常为 zip；官方包内为 image/AMBER_*.jpg，需展平到 images/ 根目录与 JSON 一致
    amber_img = ROOT / "data/benchmarks/amber/images"
    for z in amber_img.glob("*.zip"):
        with open(LOG_DIR / "parallel_orchestrator.log", "a", encoding="utf-8") as orch2:
            orch2.write(f"NOTE: extracting AMBER zip {z.name}\n")
        subprocess.run(
            ["unzip", "-q", "-o", str(z), "-d", str(amber_img)],
            cwd=str(ROOT),
            check=False,
        )
        nested = amber_img / "image"
        if nested.is_dir():
            for j in nested.glob("*.jpg"):
                dest = amber_img / j.name
                if not dest.exists():
                    j.rename(dest)
            try:
                nested.rmdir()
            except OSError:
                pass

    # 生成 Markdown 报告
    try:
        subprocess.run([PY, str(ROOT / "code/data/generate_dataset_status_report.py")], cwd=str(ROOT), check=False)
        rep = ROOT / "logs" / "dataset_download_report.md"
        if rep.exists():
            LOG_DIR.joinpath("parallel_rc_summary.txt").write_text(str(rc_map), encoding="utf-8")
            with open(rep, "a", encoding="utf-8") as f:
                f.write("\n## 并行下载退出码（最终）\n\n```\n")
                f.write(str(rc_map))
                f.write("\n```\n")
    except Exception:
        pass

    bad = [k for k, v in rc_map.items() if v != 0]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
