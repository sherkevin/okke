#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from launch_pope_chair_main_table import REMOTE_PROJECT, REMOTE_PYTHON, run_remote_bash

REMOTE_LOG_DIR = f"{REMOTE_PROJECT}/logs/baseline_full_matrix_resume"
REMOTE_RESULTS_DIR = f"{REMOTE_PROJECT}/logs/baseline_full_matrix/results"
MMBENCH_REMOTE_FILE = f"{REMOTE_PROJECT}/datasets/MMBench_EN_hf/data/dev-00000-of-00001-75b6649fb044d38b.parquet"

MODELS = (
    "qwen3-vl-8b",
    "qwen2-vl-7b",
    "qwen2.5-vl-7b",
    "instructblip-7b",
    "llava-v1.5-7b",
)
METHODS = ("base", "opera", "vcd", "dola")
POPE_SPLITS = ("random", "popular", "adversarial")
POPE_COUNT = 3000
CHAIR_COUNT = 5000
CHAIR_MAX_NEW_TOKENS = 384

# Known bad run observed to stall with corrupted outputs; skip it so the queue can continue.
SKIP_JOBS = {
    ("qwen3-vl-8b", "pope", "random", "dola"),
}


@dataclass(frozen=True)
class EvalJob:
    model: str
    dataset: str
    method: str
    split: str
    mini_test: int
    extra_args: tuple[str, ...]
    output_json: str

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _remote_mmbench_count() -> int:
    script = f"""set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{REMOTE_PYTHON} - <<'PY'
import pandas as pd
print(len(pd.read_parquet("{MMBENCH_REMOTE_FILE}")))
PY
"""
    proc = run_remote_bash(script, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to count remote MMBench rows:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return int(proc.stdout.strip().splitlines()[-1])


def build_jobs(mmbench_count: int) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    for model in MODELS:
        for split in POPE_SPLITS:
            for method in METHODS:
                jobs.append(
                    EvalJob(
                        model=model,
                        dataset="pope",
                        method=method,
                        split=split,
                        mini_test=POPE_COUNT,
                        extra_args=("--pope_split", split),
                        output_json=f"{REMOTE_RESULTS_DIR}/{model}/pope/{split}__{method}.json",
                    )
                )
        for method in METHODS:
            jobs.append(
                EvalJob(
                    model=model,
                    dataset="mmbench",
                    method=method,
                    split="default",
                    mini_test=mmbench_count,
                    extra_args=(),
                    output_json=f"{REMOTE_RESULTS_DIR}/{model}/mmbench/{method}.json",
                )
            )
        for method in METHODS:
            jobs.append(
                EvalJob(
                    model=model,
                    dataset="chair",
                    method=method,
                    split="default",
                    mini_test=CHAIR_COUNT,
                    extra_args=("--chair_max_new_tokens", str(CHAIR_MAX_NEW_TOKENS)),
                    output_json=f"{REMOTE_RESULTS_DIR}/{model}/chair/{method}.json",
                )
            )
    return jobs


def build_runner_body(*, jobs: list[EvalJob], matrix_log: str, manifest_tsv: str) -> str:
    lines: list[str] = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {REMOTE_PROJECT}",
        "source /etc/network_turbo >/dev/null 2>&1 || true",
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "export PYTHONUNBUFFERED=1",
        "export CUDA_VISIBLE_DEVICES=0",
        f'MATRIX_LOG="{matrix_log}"',
        f'MANIFEST="{manifest_tsv}"',
        'echo "[MATRIX] baseline_full_matrix_resume_gpu0 start $(date -Iseconds)"',
        'printf "iso_time\\tmodel\\tdataset\\tsplit\\tmethod\\taction\\texit_code\\tresult_json\\tmatrix_parent_log\\n" > "$MANIFEST"',
        "",
        'is_final_json() {',
        '  local path="$1"',
        f'  "{REMOTE_PYTHON}" - "$path" <<\'PYEOF\'',
        "import json",
        "import sys",
        "from pathlib import Path",
        "path = Path(sys.argv[1])",
        "if not path.exists():",
        "    raise SystemExit(1)",
        "try:",
        "    payload = json.loads(path.read_text(encoding='utf-8'))",
        "except Exception:",
        "    raise SystemExit(2)",
        "status = payload.get('status')",
        "sample_count = int(payload.get('sample_count', payload.get('n_samples', -1)) or -1)",
        "n_errors = int(payload.get('n_errors', 0) or 0)",
        "target = int(payload.get('target_samples', sample_count) or sample_count)",
        "if status == 'final' and n_errors == 0 and sample_count == target:",
        "    print('FINAL_OK')",
        "    raise SystemExit(0)",
        "raise SystemExit(3)",
        "PYEOF",
        '}',
        "",
    ]

    for job in jobs:
        skip_reason = None
        if (job.model, job.dataset, job.split, job.method) in SKIP_JOBS:
            skip_reason = "skip_known_issue"
        lines.append(f'echo "[CHECK] {job.tag} $(date -Iseconds)"')
        if skip_reason is not None:
            lines.append(f'echo "[SKIP] {job.tag} reason={skip_reason} $(date -Iseconds)"')
            lines.append(
                '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
                '"$(date -Iseconds)" '
                f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "{skip_reason}" "0" "{job.output_json}" "$MATRIX_LOG" >> "$MANIFEST"'
            )
            lines.append("")
            continue

        cmd = _shell_join(
            [
                REMOTE_PYTHON,
                f"{REMOTE_PROJECT}/run_eval_pipeline.py",
                "--model",
                job.model,
                "--dataset",
                job.dataset,
                "--method",
                job.method,
                "--mini_test",
                str(job.mini_test),
                "--output_json",
                job.output_json,
                *job.extra_args,
            ]
        )
        lines.append(f'if is_final_json "{job.output_json}" >/dev/null 2>&1; then')
        lines.append(f'  echo "[SKIP] {job.tag} reason=already_final $(date -Iseconds)"')
        lines.append(
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "skip_existing_final" "0" "{job.output_json}" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append('else')
        lines.append(f'  echo "[START] {job.tag} $(date -Iseconds)"')
        lines.append(f'  mkdir -p "{Path(job.output_json).parent.as_posix()}"')
        lines.append(f'  {cmd}')
        lines.append('  rc=$?')
        lines.append(
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" '
            '"$(date -Iseconds)" '
            f'"{job.model}" "{job.dataset}" "{job.split}" "{job.method}" "executed" "$rc" "{job.output_json}" "$MATRIX_LOG" >> "$MANIFEST"'
        )
        lines.append(f'  echo "[DONE] {job.tag} rc=$rc output={job.output_json} $(date -Iseconds)"')
        lines.append('  if [ "$rc" -ne 0 ]; then echo "[WARN] non-zero exit; continuing queue"; fi')
        lines.append('fi')
        lines.append("")

    lines.append('echo "[MATRIX] baseline_full_matrix_resume_gpu0 finished $(date -Iseconds)"')
    return "\n".join(lines) + "\n"


def remote_launch_script(*, runner_body: str, runner_sh: str, matrix_log: str) -> str:
    delim = "RUNNER_" + secrets.token_hex(16)
    if delim in runner_body:
        raise RuntimeError("heredoc delimiter collision; retry")
    return (
        "set -euo pipefail\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        f"mkdir -p {REMOTE_LOG_DIR}\n"
        f"cat > \"{runner_sh}\" <<'{delim}'\n"
        f"{runner_body}"
        f"{delim}\n"
        f"chmod +x \"{runner_sh}\"\n"
        f"nohup /bin/bash \"{runner_sh}\" >> \"{matrix_log}\" 2>&1 < /dev/null &\n"
        "echo \"LAUNCH_PID=$!\"\n"
        f"echo \"RUNNER={runner_sh}\"\n"
        f"echo \"LOG={matrix_log}\"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume the full baseline matrix on remote GPU0, skipping completed finals.")
    parser.add_argument("--mode", choices=["render", "launch"], default="launch")
    args = parser.parse_args()

    mmbench_count = _remote_mmbench_count()
    jobs = build_jobs(mmbench_count)
    ts = time.strftime("%Y%m%d_%H%M%S")
    matrix_log = f"{REMOTE_LOG_DIR}/baseline_full_matrix_resume_gpu0_{ts}.log"
    runner_sh = f"{REMOTE_LOG_DIR}/baseline_full_matrix_resume_gpu0_{ts}.sh"
    manifest_tsv = f"{REMOTE_LOG_DIR}/baseline_full_matrix_resume_gpu0_{ts}.manifest.tsv"
    body = build_runner_body(jobs=jobs, matrix_log=matrix_log, manifest_tsv=manifest_tsv)

    local_dir = Path(__file__).resolve().parent / "experiment_logs" / "remote_runners"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_copy = local_dir / f"baseline_full_matrix_resume_gpu0_{ts}.sh"
    local_copy.write_text(body, encoding="utf-8")

    if args.mode == "render":
        print(body)
        return 0

    proc = run_remote_bash(
        remote_launch_script(runner_body=body, runner_sh=runner_sh, matrix_log=matrix_log),
        timeout=300,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print("\n--- Launch Summary ---\n")
    print(
        json.dumps(
            {
                "mmbench_count": mmbench_count,
                "job_count": len(jobs),
                "skip_jobs": sorted(SKIP_JOBS),
                "local_runner": str(local_copy),
                "remote_log": matrix_log,
                "remote_runner": runner_sh,
                "remote_manifest": manifest_tsv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
