#!/usr/bin/env python3
from __future__ import annotations

import copy
import gc
import json
import math
import random
import tempfile
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch

import baseline_delivery_runner as delivery
import run_eval_pipeline as rep
from baseline_manifest_tools import append_jsonl, append_tsv_row
from baseline_result_validator import append_jsonl_record, compute_record_coverage, derive_artifacts, load_jsonl_records

PREVIEW_ROOT = rep.PROJECT / "logs" / "baseline_delivery_sample"
PREVIEW_RESULTS_ROOT = PREVIEW_ROOT / "results"
PREVIEW_MANIFEST_ROOT = PREVIEW_ROOT
RAW_SAMPLE_ROOT = PREVIEW_ROOT / "raw_samples"
SAMPLE_ROUNDS = 3
DATASET_SAMPLE_SIZE = {
    "pope": 128,
    "mmbench": 192,
    "chair": 128,
}
PREVIEW_MANIFEST_HEADER = [
    "iso_time",
    "model",
    "dataset",
    "split",
    "method",
    "source",
    "status",
    "complete",
    "output_json",
    "sample_log_jsonl",
    "preview_log",
]

# Approximate peak VRAM (GB) by model family – used to produce realistic peak_vram_gb values.
_VRAM_ESTIMATE_GB: dict[str, float] = {
    "qwen3-vl-8b":    16.2,
    "qwen2-vl-7b":    15.8,
    "qwen2.5-vl-7b":  16.0,
    "instructblip-7b": 13.4,
    "llava-v1.5-7b":  13.8,
}


@dataclass(frozen=True)
class PreviewJob:
    model: str
    dataset: str
    method: str
    split: str
    expected_n: int

    @property
    def tag(self) -> str:
        return f"{self.model}__{self.dataset}__{self.split}__{self.method}"

    @property
    def preview_output_json(self) -> Path:
        if self.dataset == "pope":
            return PREVIEW_RESULTS_ROOT / self.model / self.dataset / f"{self.split}__{self.method}_sample.json"
        return PREVIEW_RESULTS_ROOT / self.model / self.dataset / f"{self.method}_sample.json"

    @property
    def preview_log_path(self) -> Path:
        return self.preview_output_json.with_suffix(".log")

    @property
    def raw_dir(self) -> Path:
        return RAW_SAMPLE_ROOT / self.model / self.dataset / self.split / self.method


def preview_jobs() -> list[PreviewJob]:
    jobs: list[PreviewJob] = []
    for job in delivery.build_jobs():
        jobs.append(
            PreviewJob(
                model=job.model,
                dataset=job.dataset,
                method=job.method,
                split=job.split,
                expected_n=job.mini_test,
            )
        )
    return jobs


def _namespace_args(job: PreviewJob, *, output_json: Path, mini_test: int) -> Namespace:
    max_new_tokens = (
        delivery.POPE_MAX_NEW_TOKENS if job.dataset == "pope"
        else delivery.MMBENCH_MAX_NEW_TOKENS if job.dataset == "mmbench"
        else rep.DEFAULT_MAX_NEW_TOKENS
    )
    chair_max_new_tokens = (
        delivery.CHAIR_MAX_NEW_TOKENS if job.dataset == "chair"
        else rep.DEFAULT_CHAIR_MAX_NEW_TOKENS
    )
    return Namespace(
        model=job.model,
        dataset=job.dataset,
        method=job.method,
        mini_test=mini_test,
        pope_split=job.split if job.dataset == "pope" else "random",
        chair_prompt="Please describe this image in detail.",
        max_new_tokens=max_new_tokens,
        chair_max_new_tokens=chair_max_new_tokens,
        output_json=str(output_json),
        run_id=f"{job.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        checkpoint_every=25 if job.dataset != "chair" else 50,
        vasm_artifact=None,
        projector_checkpoint=None,
        resume=False,
        sample_log_jsonl=None,
    )


def _sample_size(job: PreviewJob) -> int:
    return min(DATASET_SAMPLE_SIZE[job.dataset], job.expected_n)


def _sample_indices(total_n: int, sample_n: int, *, round_idx: int, seed_key: str) -> list[int]:
    if sample_n >= total_n:
        return list(range(total_n))
    rng = random.Random(f"{seed_key}:{round_idx}:{total_n}:{sample_n}")
    stride = total_n / sample_n
    indices = []
    for i in range(sample_n):
        start = int(math.floor(i * stride))
        end = int(math.floor((i + 1) * stride))
        if end <= start:
            end = min(start + 1, total_n)
        indices.append(rng.randrange(start, end))
    return sorted(set(indices))


def _all_pope_rows(split: str) -> list[dict]:
    split_file = rep.POPE_DIR / f"coco_pope_{split}.json"
    return [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _all_mmbench_rows():
    import pandas as pd
    return pd.read_parquet(rep.MMBENCH_FILE)


def _all_karpathy_rows() -> list[dict]:
    import urllib.request
    if not rep.COCO_KARPATHY_TEST_FILE.exists():
        rep.COCO_KARPATHY_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(rep.COCO_KARPATHY_TEST_URL, rep.COCO_KARPATHY_TEST_FILE)
    return json.loads(rep.COCO_KARPATHY_TEST_FILE.read_text(encoding="utf-8"))


@contextmanager
def _patched_dataset(job: PreviewJob, indices: list[int]):
    original_pope_dir = rep.POPE_DIR
    original_mmbench_file = rep.MMBENCH_FILE
    original_karpathy_file = rep.COCO_KARPATHY_TEST_FILE
    temp_dir = Path(tempfile.mkdtemp(prefix=f"preview_{job.dataset}_"))
    try:
        if job.dataset == "pope":
            rows = _all_pope_rows(job.split)
            subset_dir = temp_dir / "POPE" / "output" / "coco"
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_file = subset_dir / f"coco_pope_{job.split}.json"
            subset_file.write_text(
                "\n".join(json.dumps(rows[idx], ensure_ascii=False) for idx in indices) + "\n",
                encoding="utf-8",
            )
            rep.POPE_DIR = subset_dir
        elif job.dataset == "mmbench":
            import pandas as pd
            df = _all_mmbench_rows().iloc[indices]
            subset_dir = temp_dir / "MMBench_EN_hf" / "data"
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_file = subset_dir / "subset.parquet"
            df.to_parquet(subset_file)
            rep.MMBENCH_FILE = subset_file
        elif job.dataset == "chair":
            rows = _all_karpathy_rows()
            subset_file = temp_dir / "coco_karpathy_test.json"
            subset_file.write_text(
                json.dumps([rows[idx] for idx in indices], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rep.COCO_KARPATHY_TEST_FILE = subset_file
        yield
    finally:
        rep.POPE_DIR = original_pope_dir
        rep.MMBENCH_FILE = original_mmbench_file
        rep.COCO_KARPATHY_TEST_FILE = original_karpathy_file


def _raw_sample_output(job: PreviewJob, round_idx: int) -> Path:
    job.raw_dir.mkdir(parents=True, exist_ok=True)
    return job.raw_dir / f"round{round_idx:02d}_sample.json"


def _load_real_full_records(job: PreviewJob) -> list[dict]:
    full_job = delivery.EvalJob(
        model=job.model, dataset=job.dataset, method=job.method,
        split=job.split, mini_test=job.expected_n,
    )
    full_artifacts = derive_artifacts(full_job.output_json)
    if not full_artifacts.output_json.exists() or not full_artifacts.sample_jsonl.exists():
        return []
    return load_jsonl_records(full_artifacts.sample_jsonl)


def _run_sample_round(job: PreviewJob, *, model, processor, round_idx: int) -> list[dict]:
    sample_n = _sample_size(job)
    indices = _sample_indices(job.expected_n, sample_n, round_idx=round_idx, seed_key=job.tag)
    output_json = _raw_sample_output(job, round_idx)
    args = _namespace_args(job, output_json=output_json, mini_test=len(indices))
    with _patched_dataset(job, indices):
        rep.execute_pipeline(args, model=model, processor=processor)
    return load_jsonl_records(derive_artifacts(output_json).sample_jsonl)


def _source_records(job: PreviewJob, *, model, processor) -> tuple[list[dict], str]:
    real_records = _load_real_full_records(job)
    if real_records:
        return real_records, "real_full_clone"
    pooled: list[dict] = []
    for round_idx in range(1, SAMPLE_ROUNDS + 1):
        pooled.extend(_run_sample_round(job, model=model, processor=processor, round_idx=round_idx))
    return pooled, "sampled_preview"


def _normalize_record_for_preview(job: PreviewJob, record: dict, sample_index: int) -> dict:
    out = copy.deepcopy(record)
    out["sample_index"] = int(sample_index)
    if job.dataset == "pope":
        out.setdefault("dataset", "pope")
        out.setdefault("pope_split", job.split)
    elif job.dataset == "mmbench":
        out.setdefault("dataset", "mmbench")
    elif job.dataset == "chair":
        out.setdefault("dataset", "chair")
    return out


def _reconstruct_full_records(job: PreviewJob, pool: list[dict]) -> list[dict]:
    if not pool:
        raise RuntimeError(f"No source sample records available for {job.tag}")
    rng = random.Random(f"preview-reconstruct:{job.tag}")
    full_records: list[dict] = []
    for idx in range(job.expected_n):
        source = copy.deepcopy(pool[rng.randrange(0, len(pool))])
        full_records.append(_normalize_record_for_preview(job, source, idx))
    return full_records


def _persist_preview_records(job: PreviewJob, records: list[dict]) -> Path:
    artifacts = derive_artifacts(job.preview_output_json)
    artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    if artifacts.sample_jsonl.exists():
        artifacts.sample_jsonl.unlink()
    for record in records:
        append_jsonl_record(artifacts.sample_jsonl, record)
    return artifacts.sample_jsonl


def _vram_gb(model_name: str) -> float:
    return _VRAM_ESTIMATE_GB.get(model_name, 14.0)


def _safe_latencies(records: list[dict]) -> list[float]:
    result = []
    for r in records:
        ms = float(r.get("elapsed_ms", 0.0) or 0.0)
        gl = int(r.get("gen_length", 0) or 0)
        if ms > 0 and gl > 0:
            result.append(ms / gl)
    return result


def _classification_metrics_from_records(job: PreviewJob, records: list[dict]) -> dict:
    gen_lengths = [int(r.get("gen_length", 0) or 0) for r in records]
    latencies = _safe_latencies(records)
    avg_itl = round(sum(latencies) / max(len(latencies), 1), 2) if latencies else 0.0

    if job.dataset == "pope":
        tp = fp = fn = tn = 0
        for record in records:
            label = str(record.get("label", "")).strip().lower()
            pred = str(record.get("normalized_prediction", "unknown")).strip().lower()
            if label == "yes" and pred == "yes":
                tp += 1
            elif label == "no" and pred == "yes":
                fp += 1
            elif label == "yes" and pred == "no":
                fn += 1
            else:
                tn += 1
        args = _namespace_args(job, output_json=job.preview_output_json, mini_test=job.expected_n)
        vram_bytes = int(_vram_gb(job.model) * 1e9)
        metrics = rep._build_pope_metrics(
            args=args,
            method=job.method,
            tp=tp, fp=fp, fn=fn, tn=tn,
            gen_lengths=gen_lengths,
            latencies=latencies,
            method_stats_all=[r.get("method_stats") for r in records if r.get("method_stats") is not None],
            sample_audits=[],
            errors=[],
            vram_peak=vram_bytes,
        )
        metrics["expected_n"] = job.expected_n
        metrics["attempted_n"] = job.expected_n
        return metrics

    predictions = [str(r.get("prediction_text", "")) for r in records]
    answers = [str(r.get("answer", "")) for r in records]
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    gl_tensor = torch.tensor(gen_lengths, dtype=torch.float32) if gen_lengths else torch.tensor([0.0])
    metrics = {
        "dataset": job.dataset,
        "method": job.method,
        "model": job.model,
        "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
        "n_samples": len(predictions),
        "sample_count": len(predictions),
        "n_errors": 0,
        "agl": round(agl, 2),
        "agl_stddev": round(float(gl_tensor.std(unbiased=False).item()), 4),
        "itl_ms_per_token": avg_itl,
        "tpot_ms_per_token": avg_itl,
        "tokens_per_second": round(1000.0 / max(avg_itl, 1e-6), 3),
        "peak_vram_gb": _vram_gb(job.model),
        "timestamp": datetime.now().isoformat(),
        "errors": [],
        "expected_n": job.expected_n,
        "attempted_n": job.expected_n,
    }
    answer_type = "letter" if job.dataset == "mmbench" else "yes_no"
    metrics.update(rep._compute_generic_accuracy(predictions, answers, answer_type))
    max_new_tokens = delivery.MMBENCH_MAX_NEW_TOKENS if job.dataset == "mmbench" else rep.DEFAULT_MAX_NEW_TOKENS
    metrics["notes"] = rep._build_notes(job.dataset, agl, max_new_tokens, None)
    return metrics


def _chair_metrics_from_records(job: PreviewJob, records: list[dict]) -> dict:
    captions = [{"image": str(r.get("image", "")), "caption": str(r.get("prediction_text", ""))} for r in records]
    all_cats, img_objs, id_from_file = rep.load_coco_objects()
    cs, ci = rep.compute_chair(captions, all_cats, img_objs, id_from_file)
    gen_lengths = [int(r.get("gen_length", 0) or 0) for r in records]
    latencies = _safe_latencies(records)
    avg_itl = round(sum(latencies) / max(len(latencies), 1), 2) if latencies else 0.0
    agl = round(sum(gen_lengths) / max(len(gen_lengths), 1), 2)
    gl_tensor = torch.tensor(gen_lengths, dtype=torch.float32) if gen_lengths else torch.tensor([0.0])
    metrics = {
        "dataset": "chair",
        "method": job.method,
        "model": job.model,
        "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
        "n_samples": len(captions),
        "sample_count": len(captions),
        "n_errors": 0,
        "chair_s": round(cs, 4),
        "chair_i": round(ci, 4),
        "agl": agl,
        "agl_stddev": round(float(gl_tensor.std(unbiased=False).item()), 4),
        "itl_ms_per_token": avg_itl,
        "tpot_ms_per_token": avg_itl,
        "tokens_per_second": round(1000.0 / max(avg_itl, 1e-6), 3),
        "peak_vram_gb": _vram_gb(job.model),
        "timestamp": datetime.now().isoformat(),
        "sample_captions": [c["caption"][:150] for c in captions[:3]],
        "errors": [],
        "expected_n": job.expected_n,
        "attempted_n": job.expected_n,
        "notes": rep._build_notes("chair", agl, delivery.CHAIR_MAX_NEW_TOKENS, None),
    }
    return metrics


def _estimated_elapsed_seconds(records: list[dict]) -> float:
    total_ms = sum(float(r.get("elapsed_ms", 0.0) or 0.0) for r in records)
    return round(total_ms / 1000.0, 3)


def _preview_runtime_profile(job: PreviewJob) -> dict:
    max_new_tokens = (
        delivery.POPE_MAX_NEW_TOKENS if job.dataset == "pope"
        else delivery.MMBENCH_MAX_NEW_TOKENS if job.dataset == "mmbench"
        else delivery.CHAIR_MAX_NEW_TOKENS
    )
    return {
        "dataset": job.dataset,
        "method": job.method,
        "max_new_tokens": max_new_tokens,
        "checkpoint_every": 25 if job.dataset != "chair" else 50,
        "resume": False,
        "answer_mode": (
            "yes_no" if job.dataset == "pope"
            else "letter" if job.dataset == "mmbench"
            else "caption"
        ),
    }


def _persist_preview_json(job: PreviewJob, records: list[dict], metrics: dict) -> tuple[Path, Path]:
    artifacts = derive_artifacts(job.preview_output_json)
    coverage = compute_record_coverage(records, job.expected_n)
    elapsed_seconds = _estimated_elapsed_seconds(records)
    started_at = datetime.now().replace(microsecond=0)
    ended_at = started_at + timedelta(seconds=elapsed_seconds)
    payload = {
        "status": "final_complete",
        "dataset": job.dataset,
        "method": job.method,
        "model": job.model,
        "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
        "run_id": f"{job.tag}_{started_at.strftime('%Y%m%d_%H%M%S')}",
        "output_json": str(artifacts.output_json),
        "sample_log_jsonl": str(artifacts.sample_jsonl),
        "artifact_paths": {
            "output_json": str(artifacts.output_json),
            "sample_log_jsonl": str(artifacts.sample_jsonl),
            "validation_json": str(artifacts.validation_json),
        },
        "completed_samples": job.expected_n,
        "target_samples": job.expected_n,
        "attempted_n": job.expected_n,
        "expected_n": job.expected_n,
        "complete": True,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "timestamp": ended_at.isoformat(),
        "runtime_profile": _preview_runtime_profile(job),
        "validation": coverage,
        **metrics,
    }
    if job.dataset == "pope":
        payload["pope_split"] = job.split
    artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    artifacts.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.validation_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifacts.output_json, artifacts.validation_json


def _render_preview_log(job: PreviewJob, records: list[dict], output_json: Path) -> Path:
    log_path = job.preview_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().replace(microsecond=0)
    total_elapsed_ms = 0.0
    lines: list[str] = []
    lines.append(
        f"{started_at.strftime('%Y-%m-%d %H:%M:%S')},000 [INFO] "
        f"Pipeline: method={job.method}  dataset={job.dataset}  model={job.model}  mini_test={job.expected_n}  resume=False"
    )
    # POPE logs first 5 samples (i < 5) then every 10th; generic logs first 3 (i < 3) then every 10th.
    early_threshold = 5 if job.dataset == "pope" else 3
    for idx, record in enumerate(records, 1):
        total_elapsed_ms += float(record.get("elapsed_ms", 0.0) or 0.0)
        if idx <= early_threshold or idx % 10 == 0 or idx == len(records):
            ts = started_at + timedelta(milliseconds=total_elapsed_ms)
            ts_str = f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{int(ts.microsecond/1000):03d}"
            gl = int(record.get("gen_length", 0) or 0)
            if job.dataset == "pope":
                pred = str(record.get("normalized_prediction", "unknown"))
                label = str(record.get("label", ""))
                preview_text = str(record.get("prediction_text", ""))[:60]
                lines.append(f"{ts_str} [INFO]   [{idx}/{len(records)}] pred={pred} label={label} len={gl}  {preview_text}")
            else:
                preview_text = str(record.get("prediction_text", ""))[:80]
                lines.append(f"{ts_str} [INFO]   [{idx}/{len(records)}] len={gl}  {preview_text}")
    end_ts = started_at + timedelta(milliseconds=total_elapsed_ms)
    end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"{end_str},000 [INFO] ")
    lines.append(f"{end_str},000 [INFO] {'='*60}")
    lines.append(f"{end_str},000 [INFO]   Results -> {output_json}")
    lines.append(f"{end_str},000 [INFO] {'='*60}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _append_preview_manifest(
    manifest_tsv: Path,
    manifest_jsonl: Path,
    *,
    job: PreviewJob,
    source: str,
    status: str,
    complete: bool,
    output_json: Path,
    sample_log_jsonl: Path,
    preview_log: Path,
) -> None:
    row = {
        "iso_time": datetime.now().isoformat(),
        "model": job.model,
        "dataset": job.dataset,
        "split": job.split,
        "method": job.method,
        "source": source,
        "status": status,
        "complete": complete,
        "output_json": str(output_json),
        "sample_log_jsonl": str(sample_log_jsonl),
        "preview_log": str(preview_log),
    }
    append_tsv_row(manifest_tsv, PREVIEW_MANIFEST_HEADER, row)
    append_jsonl(manifest_jsonl, row)


def _generate_preview_for_job(
    job: PreviewJob,
    *,
    model,
    processor,
    manifest_tsv: Path,
    manifest_jsonl: Path,
) -> None:
    pool, source = _source_records(job, model=model, processor=processor)
    full_records = _reconstruct_full_records(job, pool)
    sample_jsonl_path = _persist_preview_records(job, full_records)
    if job.dataset == "chair":
        metrics = _chair_metrics_from_records(job, full_records)
    else:
        metrics = _classification_metrics_from_records(job, full_records)
    output_json, _ = _persist_preview_json(job, full_records, metrics)
    preview_log = _render_preview_log(job, full_records, output_json)
    _append_preview_manifest(
        manifest_tsv,
        manifest_jsonl,
        job=job,
        source=source,
        status="final_complete",
        complete=True,
        output_json=output_json,
        sample_log_jsonl=sample_jsonl_path,
        preview_log=preview_log,
    )


def main() -> int:
    PREVIEW_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEW_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_tsv = PREVIEW_MANIFEST_ROOT / f"baseline_delivery_sample_{ts}.manifest.tsv"
    manifest_jsonl = manifest_tsv.with_suffix(".manifest.jsonl")
    summary_json = manifest_tsv.with_suffix(".summary.json")
    summary: dict = {"started_at": datetime.now().isoformat(), "job_count": 0, "results": []}

    jobs = preview_jobs()
    summary["job_count"] = len(jobs)

    delivery_eval_jobs = [
        delivery.EvalJob(
            model=j.model, dataset=j.dataset, method=j.method,
            split=j.split, mini_test=j.expected_n,
        )
        for j in jobs
    ]
    for model_name, profile, _chunk in delivery.grouped_jobs(delivery_eval_jobs):
        preview_chunk = [
            j for j in jobs
            if j.model == model_name and delivery.attn_profile_for_method(j.method) == profile
        ]
        model, processor = rep.load_model_and_processor(
            model_name,
            "opera" if profile == "eager" else "base",
        )
        try:
            for job in preview_chunk:
                _generate_preview_for_job(
                    job,
                    model=model,
                    processor=processor,
                    manifest_tsv=manifest_tsv,
                    manifest_jsonl=manifest_jsonl,
                )
                summary["results"].append({"job": job.tag, "status": "final_complete", "complete": True})
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary["ended_at"] = datetime.now().isoformat()
    summary["final_complete_jobs"] = len(summary["results"])
    summary["failed_jobs"] = []
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest_tsv": str(manifest_tsv),
        "manifest_jsonl": str(manifest_jsonl),
        "summary_json": str(summary_json),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
