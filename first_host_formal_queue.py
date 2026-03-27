#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(r"d:\Shervin\OneDrive\Desktop\breaking")
LOCAL_RESULTS_DIR = WORKSPACE / "experiment_logs" / "uniground_v6"
LOCAL_FEED = LOCAL_RESULTS_DIR / "first_host_batch_ready_20260322.md"

SSH_KEY = r"C:\Users\shers\.ssh\id_ed25519_autodl"
SSH_PORT = "47559"
SSH_HOST = "root@connect.westc.seetacloud.com"

REMOTE_PROJECT = "/root/autodl-tmp/BRA_Project"
REMOTE_RESULTS_DIR = f"{REMOTE_PROJECT}/logs/uniground_v6"
REMOTE_FEED = f"{REMOTE_RESULTS_DIR}/first_host_batch_ready_20260322.md"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_ENCODER = f"{REMOTE_PROJECT}/models/clip-vit-large-patch14"
REMOTE_CHECKPOINT = f"{REMOTE_PROJECT}/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt"
REMOTE_PROJECTOR = f"{REMOTE_PROJECT}/models/V_matrix.pt"

HOST_MODEL = "Qwen3-VL-8B-Instruct"
RUNNER_MODEL_ARG = "qwen3-vl-8b"
POPE_FORMAL_COUNT = 3000
CHAIR_FORMAL_COUNT = 150
CHAIR_MAX_NEW_TOKENS = 384

POPE_METHODS = [
    "base",
    "tlra_internal_zero",
    "tlra_internal_calib",
    "external_global_prior",
    "uniground",
    "uniground_no_gate",
    "uniground_global_only",
    "uniground_no_abstain",
]

CHAIR_METHODS = [
    "base",
    "tlra_internal_zero",
    "tlra_internal_calib",
    "external_global_prior",
    "uniground",
]


def run_checked(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs = {"check": True, "text": True}
    if capture:
        kwargs["capture_output"] = True
    return subprocess.run(cmd, **kwargs)


def ssh_base() -> list[str]:
    return [
        "ssh",
        "-i",
        SSH_KEY,
        "-p",
        SSH_PORT,
        "-o",
        "StrictHostKeyChecking=no",
        SSH_HOST,
    ]


def scp_base() -> list[str]:
    return [
        "scp",
        "-i",
        SSH_KEY,
        "-P",
        SSH_PORT,
        "-o",
        "StrictHostKeyChecking=no",
    ]


def remote_command(dataset: str, methods: list[str]) -> str:
    eval_parts = [
        "CUDA_VISIBLE_DEVICES=1",
        REMOTE_PYTHON,
        "run_uniground_eval.py",
        "--model",
        RUNNER_MODEL_ARG,
        "--dataset",
        dataset,
        "--methods",
        *methods,
        "--projector_checkpoint",
        REMOTE_PROJECTOR,
        "--psi_checkpoint",
        REMOTE_CHECKPOINT,
        "--external_device",
        "cpu",
        "--external_encoder",
        REMOTE_ENCODER,
        "--output_dir",
        REMOTE_RESULTS_DIR,
    ]
    if dataset == "pope":
        eval_parts.extend(["--pope_split", "random", "--mini_test", str(POPE_FORMAL_COUNT)])
    else:
        eval_parts.extend(
            [
                "--mini_test",
                str(CHAIR_FORMAL_COUNT),
                "--chair_max_new_tokens",
                str(CHAIR_MAX_NEW_TOKENS),
            ]
        )
    return (
        "source /etc/network_turbo >/dev/null 2>&1 || true; "
        f"cd {REMOTE_PROJECT} && "
        "export PATH=/root/miniconda3/bin:$PATH && "
        "export PYTHONUNBUFFERED=1 && "
        + " ".join(eval_parts)
    )


def run_batch(dataset: str, methods: list[str]) -> list[str]:
    cmd = ssh_base() + [remote_command(dataset, methods)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    remote_paths: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        stripped = line.strip()
        if stripped.endswith(".json") and REMOTE_RESULTS_DIR in stripped:
            remote_paths.append(stripped)
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    deduped: list[str] = []
    seen = set()
    for path in remote_paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    if len(deduped) != len(methods):
        raise RuntimeError(f"Expected {len(methods)} JSON outputs, got {len(deduped)}: {deduped}")
    return deduped


def mirror_results(remote_paths: list[str]) -> None:
    cmd = scp_base() + [f"{SSH_HOST}:{path}" for path in remote_paths] + [str(LOCAL_RESULTS_DIR)]
    run_checked(cmd)


def append_feed(dataset: str, methods: list[str], remote_paths: list[str]) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    block = [
        "",
        "## Batch",
        "",
        f"- timestamp: `{timestamp}`",
        f"- host model: `{HOST_MODEL}`",
        f"- methods completed: {', '.join(f'`{method}`' for method in methods)}",
        f"- dataset: `{dataset}`",
        "- exact JSON filenames:",
    ]
    block.extend([f"  - `{Path(path).name}`" for path in remote_paths])
    block.append("- local mirror status: mirrored to `experiment_logs/uniground_v6/` and verified present locally")
    with LOCAL_FEED.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))
        f.write("\n")


def sync_feed() -> None:
    cmd = scp_base() + [str(LOCAL_FEED), f"{SSH_HOST}:{REMOTE_FEED}"]
    run_checked(cmd)


def main() -> None:
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, methods in (("pope", POPE_METHODS), ("chair", CHAIR_METHODS)):
        remote_paths = run_batch(dataset, methods)
        mirror_results(remote_paths)
        append_feed(dataset, methods, remote_paths)
        sync_feed()


if __name__ == "__main__":
    main()
