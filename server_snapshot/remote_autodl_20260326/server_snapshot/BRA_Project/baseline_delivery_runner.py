#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

import run_eval_pipeline as rep
from baseline_manifest_tools import append_jsonl, append_tsv_row
from baseline_result_validator import derive_artifacts, validate_job_output

RESULT_ROOT = rep.PROJECT / "logs" / "baseline_delivery" / "results"
MANIFEST_ROOT = rep.PROJECT / "logs" / "baseline_delivery"
MODELS = (
    "qwen3-vl-8b",
    "qwen2-vl-7b",
    "qwen2.5-vl-7b",
    "instructblip-7b",
    "llava-v1.5-7b",
)
METHOD_ORDER = ("base", "vcd", "dola", "opera")
POPE_SPLITS = ("random", "popular", "adversarial")
POPE_COUNT = 3000
CHAIR_COUNT = 5000
POPE_MAX_NEW_TOKENS = 8
MMBENCH_MAX_NEW_TOKENS = 4
CHAIR_MAX_NEW_TOKENS = 96
RETRYABLE_DOLA_LANES = {
    ("qwen3-vl-8b", "pope", "random", "dola"),
}
MANIFEST_HEADER = [
    "iso_time",
    "model",
    "dataset",
    "split",
    "method",
    "action",
    "attempt",
    "status",
    "complete",
    "fallback_policy",
    "issues",
    "output_json",
    "sample_log_jsonl",
]


@dataclass(frozen=True)
class EvalJob:
    model: str
    dataset: str
    method: str
    split: str
    mini_test: int

    @property
    def output_json(self) -> Path:
        if self.dataset == "pope":
            return RESULT_ROOT / self.model / self.dataset / f"{self.split}__{self.method}.json"
        return RESULT_ROOT / self.model / self.dataset / f"{self.method}.json"

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"


def build_jobs() -> list[EvalJob]:
    import pandas as pd

    mmbench_count = len(pd.read_parquet(rep.MMBENCH_FILE, columns=[]))
    jobs: list[EvalJob] = []
    for model in MODELS:
        for method in METHOD_ORDER:
            for split in POPE_SPLITS:
                jobs.append(EvalJob(model=model, dataset="pope", method=method, split=split, mini_test=POPE_COUNT))
            jobs.append(EvalJob(model=model, dataset="mmbench", method=method, split="default", mini_test=mmbench_count))
            jobs.append(EvalJob(model=model, dataset="chair", method=method, split="default", mini_test=CHAIR_COUNT))
    return jobs


def attn_profile_for_method(method: str) -> str:
    return "eager" if method == "opera" else "sdpa"


def grouped_jobs(jobs: list[EvalJob]) -> list[tuple[str, str, list[EvalJob]]]:
    grouped: list[tuple[str, str, list[EvalJob]]] = []
    for model in MODELS:
        for profile in ("sdpa", "eager"):
            chunk = [job for job in jobs if job.model == model and attn_profile_for_method(job.method) == profile]
            if chunk:
                grouped.append((model, profile, chunk))
    return grouped


def _delete_job_artifacts(job: EvalJob) -> None:
    artifacts = derive_artifacts(job.output_json)
    for path in (artifacts.output_json, artifacts.sample_jsonl, artifacts.validation_json):
        if path.exists():
            path.unlink()


def _should_resume(job: EvalJob) -> bool:
    validation = validate_job_output(job.output_json)
    coverage = validation.get("coverage") or {}
    return (
        validation.get("output_json_exists")
        and validation.get("sample_jsonl_exists")
        and not coverage.get("duplicate_indices")
        and int(coverage.get("error_n", 0) or 0) == 0
        and bool(coverage.get("missing_indices"))
    )


def _runtime_args_for_job(job: EvalJob, *, fallback: str = "none", resume: bool = False) -> Namespace:
    max_new_tokens = rep.DEFAULT_MAX_NEW_TOKENS
    chair_max_new_tokens = rep.DEFAULT_CHAIR_MAX_NEW_TOKENS
    if job.dataset == "pope":
        max_new_tokens = 4 if fallback == "short_yesno_retry" else POPE_MAX_NEW_TOKENS
    elif job.dataset == "mmbench":
        max_new_tokens = 3 if fallback == "short_yesno_retry" else MMBENCH_MAX_NEW_TOKENS
    elif job.dataset == "chair":
        chair_max_new_tokens = 80 if fallback == "short_yesno_retry" else CHAIR_MAX_NEW_TOKENS
        max_new_tokens = min(max_new_tokens, chair_max_new_tokens)
    return Namespace(
        model=job.model,
        dataset=job.dataset,
        method=job.method,
        mini_test=job.mini_test,
        pope_split=job.split if job.dataset == "pope" else "random",
        chair_prompt="Please describe this image in detail.",
        max_new_tokens=max_new_tokens,
        chair_max_new_tokens=chair_max_new_tokens,
        output_json=str(job.output_json),
        run_id=f"delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        checkpoint_every=25 if job.dataset != "chair" else 50,
        vasm_artifact=None,
        projector_checkpoint=None,
        resume=resume,
        sample_log_jsonl=None,
    )


def _append_manifest(manifest_tsv: Path, manifest_jsonl: Path, *, job: EvalJob, action: str, attempt: int, status: str, complete: bool, fallback_policy: str, issues: list[str], output_json: Path, sample_log_jsonl: Path) -> None:
    row = {
        "iso_time": datetime.now().isoformat(),
        "model": job.model,
        "dataset": job.dataset,
        "split": job.split,
        "method": job.method,
        "action": action,
        "attempt": attempt,
        "status": status,
        "complete": complete,
        "fallback_policy": fallback_policy,
        "issues": ",".join(issues),
        "output_json": str(output_json),
        "sample_log_jsonl": str(sample_log_jsonl),
    }
    append_tsv_row(manifest_tsv, MANIFEST_HEADER, row)
    append_jsonl(manifest_jsonl, row)


def _load_model_for_profile(model_name: str, profile: str):
    rep.logger.info("Loading group model=%s profile=%s", model_name, profile)
    representative_method = "opera" if profile == "eager" else "base"
    return rep.load_model_and_processor(model_name, representative_method)


def _run_job(job: EvalJob, *, model, processor, manifest_tsv: Path, manifest_jsonl: Path) -> dict:
    artifacts = derive_artifacts(job.output_json)
    validation = validate_job_output(job.output_json)
    if not validation.get("issues"):
        _append_manifest(
            manifest_tsv,
            manifest_jsonl,
            job=job,
            action="skip_existing_final",
            attempt=0,
            status=str(validation.get("status")),
            complete=bool(validation.get("complete")),
            fallback_policy="none",
            issues=[],
            output_json=artifacts.output_json,
            sample_log_jsonl=artifacts.sample_jsonl,
        )
        return {"status": "skipped_with_reason", "complete": True}

    should_resume = _should_resume(job)
    fallback_policy = "none"
    if not should_resume and artifacts.output_json.exists():
        _delete_job_artifacts(job)

    for attempt in (1, 2):
        if attempt == 2:
            fallback_policy = "short_yesno_retry" if (job.model, job.dataset, job.split, job.method) in RETRYABLE_DOLA_LANES or job.dataset in {"pope", "mmbench"} else "retry_clean"
            _delete_job_artifacts(job)
            should_resume = False
        args = _runtime_args_for_job(job, fallback=fallback_policy, resume=should_resume)
        result = rep.execute_pipeline(args, model=model, processor=processor)
        validation = validate_job_output(job.output_json)
        _append_manifest(
            manifest_tsv,
            manifest_jsonl,
            job=job,
            action="executed",
            attempt=attempt,
            status=str(result["status"]),
            complete=bool(result["complete"]),
            fallback_policy=fallback_policy,
            issues=list(validation.get("issues", [])),
            output_json=artifacts.output_json,
            sample_log_jsonl=artifacts.sample_jsonl,
        )
        if not validation.get("issues"):
            return {"status": "final_complete", "complete": True}
        should_resume = False
    return {"status": "failed_validated", "complete": False, "issues": validation.get("issues", [])}


def main() -> int:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_tsv = MANIFEST_ROOT / f"baseline_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.manifest.tsv"
    manifest_jsonl = manifest_tsv.with_suffix(".manifest.jsonl")
    summary_json = manifest_tsv.with_suffix(".summary.json")
    jobs = build_jobs()
    summary = {
        "started_at": datetime.now().isoformat(),
        "job_count": len(jobs),
        "results": [],
    }

    for model_name, profile, chunk in grouped_jobs(jobs):
        model, processor = _load_model_for_profile(model_name, profile)
        try:
            for job in chunk:
                summary["results"].append({"job": job.tag, **_run_job(job, model=model, processor=processor, manifest_tsv=manifest_tsv, manifest_jsonl=manifest_jsonl)})
        finally:
            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary["ended_at"] = datetime.now().isoformat()
    summary["final_complete_jobs"] = sum(1 for item in summary["results"] if item.get("complete"))
    summary["failed_jobs"] = [item for item in summary["results"] if not item.get("complete")]
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_tsv": str(manifest_tsv), "manifest_jsonl": str(manifest_jsonl), "summary_json": str(summary_json)}, ensure_ascii=False, indent=2))
    return 0 if not summary["failed_jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
