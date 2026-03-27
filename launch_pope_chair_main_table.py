#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


WORKSPACE = Path(r"d:\Shervin\OneDrive\Desktop\breaking")
SSH_KEY = r"C:\Users\shers\.ssh\id_ed25519_autodl"
SSH_PORT = "47559"
SSH_HOST = "root@connect.westc.seetacloud.com"

REMOTE_PROJECT = "/root/autodl-tmp/BRA_Project"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_RESULTS_DIR = f"{REMOTE_PROJECT}/logs/main_table"
REMOTE_UNIGROUND_OUTPUT_DIR = f"{REMOTE_RESULTS_DIR}/uniground"
REMOTE_BASELINE_OUTPUT_DIR = f"{REMOTE_RESULTS_DIR}/baseline"
REMOTE_PSI_CHECKPOINT = f"{REMOTE_PROJECT}/models/uniground_v6/psi_univ_clip_large_coco_val2014_v1.pt"
REMOTE_PROJECTOR = f"{REMOTE_PROJECT}/models/V_matrix.pt"
REMOTE_ENCODER = f"{REMOTE_PROJECT}/models/clip-vit-large-patch14"


@dataclass(frozen=True)
class EvalJob:
    name: str
    dataset: str
    runner: str
    args: list[str]


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


def run_local(cmd: list[str], *, input_text: str | None = None, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
        encoding="utf-8",
        errors="replace",
    )


def run_remote_bash(script: str, *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ssh_base() + ["/bin/bash", "-s"]
    normalized = script.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    proc = subprocess.run(
        cmd,
        input=normalized,
        capture_output=True,
        timeout=timeout,
        check=check,
    )
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=proc.stdout.decode("utf-8", errors="replace"),
        stderr=proc.stderr.decode("utf-8", errors="replace"),
    )


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_jobs(*, model: str, pope_count: int, chair_count: int, include_internal_controls: bool) -> list[EvalJob]:
    jobs = [
        EvalJob(
            name="pope_baselines",
            dataset="pope",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "pope",
                "--mini_test",
                str(pope_count),
                "--pope_split",
                "random",
                "--method",
                "base",
            ],
        ),
        EvalJob(
            name="pope_vcd",
            dataset="pope",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "pope",
                "--mini_test",
                str(pope_count),
                "--pope_split",
                "random",
                "--method",
                "vcd",
            ],
        ),
        EvalJob(
            name="pope_dola",
            dataset="pope",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "pope",
                "--mini_test",
                str(pope_count),
                "--pope_split",
                "random",
                "--method",
                "dola",
            ],
        ),
        EvalJob(
            name="pope_opera",
            dataset="pope",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "pope",
                "--mini_test",
                str(pope_count),
                "--pope_split",
                "random",
                "--method",
                "opera",
            ],
        ),
        EvalJob(
            name="pope_uniground_family",
            dataset="pope",
            runner="run_uniground_eval.py",
            args=[
                "--model",
                model,
                "--dataset",
                "pope",
                "--mini_test",
                str(pope_count),
                "--pope_split",
                "random",
                "--external_encoder",
                REMOTE_ENCODER,
                "--external_device",
                "cpu",
                "--psi_checkpoint",
                REMOTE_PSI_CHECKPOINT,
                "--projector_checkpoint",
                REMOTE_PROJECTOR,
                "--output_dir",
                REMOTE_UNIGROUND_OUTPUT_DIR,
                "--methods",
                "external_global_prior",
                "uniground",
            ],
        ),
        EvalJob(
            name="chair_base",
            dataset="chair",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "chair",
                "--mini_test",
                str(chair_count),
                "--chair_max_new_tokens",
                "384",
                "--method",
                "base",
            ],
        ),
        EvalJob(
            name="chair_vcd",
            dataset="chair",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "chair",
                "--mini_test",
                str(chair_count),
                "--chair_max_new_tokens",
                "384",
                "--method",
                "vcd",
            ],
        ),
        EvalJob(
            name="chair_dola",
            dataset="chair",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "chair",
                "--mini_test",
                str(chair_count),
                "--chair_max_new_tokens",
                "384",
                "--method",
                "dola",
            ],
        ),
        EvalJob(
            name="chair_opera",
            dataset="chair",
            runner="run_eval_pipeline.py",
            args=[
                "--model",
                model,
                "--dataset",
                "chair",
                "--mini_test",
                str(chair_count),
                "--chair_max_new_tokens",
                "384",
                "--method",
                "opera",
            ],
        ),
        EvalJob(
            name="chair_uniground_family",
            dataset="chair",
            runner="run_uniground_eval.py",
            args=[
                "--model",
                model,
                "--dataset",
                "chair",
                "--mini_test",
                str(chair_count),
                "--chair_max_new_tokens",
                "384",
                "--external_encoder",
                REMOTE_ENCODER,
                "--external_device",
                "cpu",
                "--psi_checkpoint",
                REMOTE_PSI_CHECKPOINT,
                "--projector_checkpoint",
                REMOTE_PROJECTOR,
                "--output_dir",
                REMOTE_UNIGROUND_OUTPUT_DIR,
                "--methods",
                "external_global_prior",
                "uniground",
            ],
        ),
    ]
    if include_internal_controls:
        jobs.extend(
            [
                EvalJob(
                    name="pope_internal_controls",
                    dataset="pope",
                    runner="run_uniground_eval.py",
                    args=[
                        "--model",
                        model,
                        "--dataset",
                        "pope",
                        "--mini_test",
                        str(pope_count),
                        "--pope_split",
                        "random",
                        "--external_encoder",
                        REMOTE_ENCODER,
                        "--external_device",
                        "cpu",
                        "--psi_checkpoint",
                        REMOTE_PSI_CHECKPOINT,
                        "--projector_checkpoint",
                        REMOTE_PROJECTOR,
                        "--output_dir",
                        REMOTE_UNIGROUND_OUTPUT_DIR,
                        "--methods",
                        "tlra_internal_zero",
                        "tlra_internal_calib",
                    ],
                ),
                EvalJob(
                    name="chair_internal_controls",
                    dataset="chair",
                    runner="run_uniground_eval.py",
                    args=[
                        "--model",
                        model,
                        "--dataset",
                        "chair",
                        "--mini_test",
                        str(chair_count),
                        "--chair_max_new_tokens",
                        "384",
                        "--external_encoder",
                        REMOTE_ENCODER,
                        "--external_device",
                        "cpu",
                        "--psi_checkpoint",
                        REMOTE_PSI_CHECKPOINT,
                        "--projector_checkpoint",
                        REMOTE_PROJECTOR,
                        "--output_dir",
                        REMOTE_UNIGROUND_OUTPUT_DIR,
                        "--methods",
                        "tlra_internal_zero",
                        "tlra_internal_calib",
                    ],
                ),
            ]
        )
    return jobs


def job_command(job: EvalJob) -> str:
    cmd = [REMOTE_PYTHON, f"{REMOTE_PROJECT}/{job.runner}", *job.args]
    return shell_join(cmd)


def preflight_script(args: argparse.Namespace) -> str:
    checks = [
        ("project", REMOTE_PROJECT),
        ("run_eval_pipeline", f"{REMOTE_PROJECT}/run_eval_pipeline.py"),
        ("run_uniground_eval", f"{REMOTE_PROJECT}/run_uniground_eval.py"),
        ("qwen_model", f"{REMOTE_PROJECT}/models/{args.remote_model_dir}"),
        ("clip_encoder", REMOTE_ENCODER),
        ("psi_checkpoint", REMOTE_PSI_CHECKPOINT),
        ("projector", REMOTE_PROJECTOR),
        ("coco_val", f"{REMOTE_PROJECT}/datasets/coco2014/val2014"),
        ("instances_val", f"{REMOTE_PROJECT}/datasets/coco2014/annotations/instances_val2014.json"),
        ("captions_val", f"{REMOTE_PROJECT}/datasets/coco2014/annotations/captions_val2014.json"),
        ("pope_random", f"{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_random.json"),
        ("pope_popular", f"{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_popular.json"),
        ("pope_adversarial", f"{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_adversarial.json"),
    ]
    py = textwrap.dedent(
        f"""\
        from pathlib import Path
        targets = {checks!r}
        for name, raw in targets:
            p = Path(raw)
            print(f"CHECK\\t{{name}}\\t{{p.exists()}}\\t{{p.is_dir() if p.exists() else False}}")
        """
    ).rstrip()
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"{REMOTE_PYTHON} - <<'PY'\n"
        f"{py}\n"
        "PY\n"
    )


def detached_self_test_script() -> str:
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        source /etc/network_turbo >/dev/null 2>&1 || true
        export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
        mkdir -p {REMOTE_RESULTS_DIR}
        TEST_LOG="{REMOTE_RESULTS_DIR}/selftest.log"
        TEST_SH="{REMOTE_RESULTS_DIR}/selftest.sh"
        cat > "$TEST_SH" <<'EOF'
        #!/bin/bash
        set -euo pipefail
        echo SELFTEST_START
        sleep 1
        echo SELFTEST_DONE
        EOF
        chmod +x "$TEST_SH"
        nohup /bin/bash "$TEST_SH" > "$TEST_LOG" 2>&1 < /dev/null &
        PID=$!
        echo "PID=$PID"
        for _ in $(seq 1 10); do
          if ! kill -0 "$PID" 2>/dev/null; then
            break
          fi
          sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then
          echo "SELFTEST_STATUS=timeout"
          exit 1
        fi
        echo "SELFTEST_STATUS=completed"
        echo "SELFTEST_LOG_PATH=$TEST_LOG"
        cat "$TEST_LOG"
        """
    )


def launch_script(args: argparse.Namespace, jobs: list[EvalJob]) -> tuple[str, str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    remote_log = f"{REMOTE_RESULTS_DIR}/main_table_{args.model}_{timestamp}.log"
    remote_runner = f"{REMOTE_RESULTS_DIR}/main_table_{args.model}_{timestamp}.sh"
    commands = []
    for job in jobs:
        commands.append(f'echo "[START] {job.name}"')
        commands.append(job_command(job))
        commands.append(f'echo "[DONE] {job.name}"')
    body = "\n".join(commands)
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        source /etc/network_turbo >/dev/null 2>&1 || true
        export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
        mkdir -p {REMOTE_RESULTS_DIR} {REMOTE_BASELINE_OUTPUT_DIR} {REMOTE_UNIGROUND_OUTPUT_DIR}
        cat > "{remote_runner}" <<'EOF'
        #!/bin/bash
        set -euo pipefail
        cd {REMOTE_PROJECT}
        export PYTHONUNBUFFERED=1
        export CUDA_VISIBLE_DEVICES={args.gpu}
        {body}
        EOF
        chmod +x "{remote_runner}"
        nohup /bin/bash "{remote_runner}" > "{remote_log}" 2>&1 < /dev/null &
        echo "LAUNCH_PID=$!"
        echo "RUNNER={remote_runner}"
        echo "LOG={remote_log}"
        """
    )
    return script, remote_runner, remote_log


def parse_launch_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust one-click launcher for POPE + CHAIR main-table baselines.")
    parser.add_argument("--mode", choices=["preflight", "selftest", "launch", "dry-run"], default="preflight")
    parser.add_argument("--model", choices=["qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b"], default="qwen3-vl-8b")
    parser.add_argument("--remote-model-dir", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--pope-count", type=int, default=3000)
    parser.add_argument("--chair-count", type=int, default=150)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--include-internal-controls", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(
        model=args.model,
        pope_count=args.pope_count,
        chair_count=args.chair_count,
        include_internal_controls=args.include_internal_controls,
    )

    if args.mode == "dry-run":
        payload = {
            "jobs": [
                {"name": job.name, "runner": job.runner, "dataset": job.dataset, "command": job_command(job)}
                for job in jobs
            ]
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "preflight":
        proc = run_remote_bash(preflight_script(args), timeout=120, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            return proc.returncode
        failed = [line for line in proc.stdout.splitlines() if line.startswith("CHECK\t") and "\tFalse\t" in line]
        return 1 if failed else 0

    if args.mode == "selftest":
        proc = run_remote_bash(detached_self_test_script(), timeout=120, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    launch_body, _runner, _log = launch_script(args, jobs)
    proc = run_remote_bash(launch_body, timeout=120, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
