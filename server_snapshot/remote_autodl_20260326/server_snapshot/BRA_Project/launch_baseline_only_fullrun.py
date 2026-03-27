#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from launch_pope_chair_main_table import run_remote_bash


REMOTE_PROJECT = "/root/autodl-tmp/BRA_Project"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_RESULTS_DIR = f"{REMOTE_PROJECT}/logs/baseline_main_table"
DEFAULT_POPE_COUNT = 3000
DEFAULT_CHAIR_COUNT = 40504
DEFAULT_METHODS = ("base", "beam_search", "vcd", "dola", "opera")
MODEL_REMOTE_DIRS = {
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    "qwen3-vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3-vl-2b": "Qwen3-VL-2B-Instruct",
    "llava-v1.5-7b": "llava-1.5-7b-hf",
    "instructblip-7b": "instructblip-vicuna-7b",
}


@dataclass(frozen=True)
class EvalJob:
    name: str
    dataset: str
    method: str
    args: list[str]


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_jobs(*, model: str, pope_split: str, pope_count: int, chair_count: int, methods: list[str], datasets: list[str]) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    if "pope" in datasets:
        for method in methods:
            jobs.append(
                EvalJob(
                    name=f"pope_{method}",
                    dataset="pope",
                    method=method,
                    args=[
                        "--model",
                        model,
                        "--dataset",
                        "pope",
                        "--method",
                        method,
                        "--mini_test",
                        str(pope_count),
                        "--pope_split",
                        pope_split,
                    ],
                )
            )
    if "chair" in datasets:
        for method in methods:
            jobs.append(
                EvalJob(
                    name=f"chair_{method}",
                    dataset="chair",
                    method=method,
                    args=[
                        "--model",
                        model,
                        "--dataset",
                        "chair",
                        "--method",
                        method,
                        "--mini_test",
                        str(chair_count),
                        "--chair_max_new_tokens",
                        "384",
                    ],
                )
            )
    return jobs


def job_command(job: EvalJob) -> str:
    return shell_join([REMOTE_PYTHON, f"{REMOTE_PROJECT}/run_eval_pipeline.py", *job.args])


def preflight_script(args: argparse.Namespace) -> str:
    remote_model_dir = args.remote_model_dir or MODEL_REMOTE_DIRS[args.model]
    py = textwrap.dedent(
        f"""\
        from pathlib import Path
        import json

        summary = {{
            "run_eval_pipeline": Path("{REMOTE_PROJECT}/run_eval_pipeline.py").exists(),
            "baseline_processors": Path("{REMOTE_PROJECT}/baseline_processors.py").exists(),
            "coco_val_dir": Path("{REMOTE_PROJECT}/datasets/coco2014/val2014").exists(),
            "instances_val": Path("{REMOTE_PROJECT}/datasets/coco2014/annotations/instances_val2014.json").exists(),
            "captions_val": Path("{REMOTE_PROJECT}/datasets/coco2014/annotations/captions_val2014.json").exists(),
            "pope_random": Path("{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_random.json").exists(),
            "pope_popular": Path("{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_popular.json").exists(),
            "pope_adversarial": Path("{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_adversarial.json").exists(),
            "model_dir": Path("{REMOTE_PROJECT}/models/{remote_model_dir}").exists(),
            "coco_val_count": len(list(Path("{REMOTE_PROJECT}/datasets/coco2014/val2014").glob("*.jpg"))),
            "pope_count": sum(1 for _ in open("{REMOTE_PROJECT}/datasets/POPE/output/coco/coco_pope_{args.pope_split}.json", encoding="utf-8")),
        }}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        """
    ).rstrip()
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"{REMOTE_PYTHON} - <<'PY'\n{py}\nPY\n"
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
        echo BASELINE_SELFTEST_START
        sleep 1
        echo BASELINE_SELFTEST_DONE
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


def launch_script(args: argparse.Namespace, jobs: list[EvalJob]) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    remote_log = f"{REMOTE_RESULTS_DIR}/baseline_fullrun_{args.model}_{timestamp}.log"
    remote_runner = f"{REMOTE_RESULTS_DIR}/baseline_fullrun_{args.model}_{timestamp}.sh"
    commands = []
    for job in jobs:
        commands.append(f'echo "[START] {job.name}"')
        commands.append(job_command(job))
        commands.append(f'echo "[DONE] {job.name}"')
    body = "\n".join(commands)
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_RESULTS_DIR}\n"
        f"cat > \"{remote_runner}\" <<'EOF'\n"
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"cd {REMOTE_PROJECT}\n"
        "export PYTHONUNBUFFERED=1\n"
        f"export CUDA_VISIBLE_DEVICES={args.gpu}\n"
        f"{body}\n"
        "EOF\n"
        f"chmod +x \"{remote_runner}\"\n"
        f"nohup /bin/bash \"{remote_runner}\" > \"{remote_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
        f"echo \"RUNNER={remote_runner}\"\n"
        f"echo \"LOG={remote_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust one-click launcher for baseline-only POPE + CHAIR runs.")
    parser.add_argument("--mode", choices=["preflight", "selftest", "launch", "dry-run"], default="preflight")
    parser.add_argument(
        "--model",
        choices=["qwen3-vl-8b", "qwen3-vl-4b", "qwen3-vl-2b", "llava-v1.5-7b", "instructblip-7b"],
        default="qwen3-vl-8b",
    )
    parser.add_argument("--remote-model-dir", default=None)
    parser.add_argument("--pope-split", choices=["random", "popular", "adversarial"], default="random")
    parser.add_argument("--pope-count", type=int, default=DEFAULT_POPE_COUNT)
    parser.add_argument("--chair-count", type=int, default=DEFAULT_CHAIR_COUNT)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--datasets", nargs="+", default=["pope", "chair"], choices=["pope", "chair"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    jobs = build_jobs(
        model=args.model,
        pope_split=args.pope_split,
        pope_count=args.pope_count,
        chair_count=args.chair_count,
        methods=args.methods,
        datasets=args.datasets,
    )

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "jobs": [
                        {
                            "name": job.name,
                            "dataset": job.dataset,
                            "method": job.method,
                            "command": job_command(job),
                        }
                        for job in jobs
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.mode == "preflight":
        proc = run_remote_bash(preflight_script(args), timeout=120, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    if args.mode == "selftest":
        proc = run_remote_bash(detached_self_test_script(), timeout=120, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode

    proc = run_remote_bash(launch_script(args, jobs), timeout=120, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
