#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path

import torch

import run_eval_pipeline as rep
from baseline_delivery_runner import EvalJob, build_jobs as build_full_jobs, grouped_jobs as group_full_jobs
from baseline_manifest_tools import append_jsonl, append_tsv_row
from baseline_result_validator import append_jsonl_record, compute_record_coverage, derive_artifacts, load_jsonl_records, validate_job_output

PREVIEW_ROOT = rep.PROJECT / "logs" / "baseline_sample_preview"
PREVIEW_RESULTS_ROOT = PREVIEW_ROOT / "results"
PREVIEW_LOG_ROOT = PREVIEW_ROOT / "job_logs"
SOURCE_RESULTS_ROOT = PREVIEW_ROOT / "source_results"
PREVIEW_MANIFEST_HEADER = [
    "iso_time",
    "model",
    "dataset",
    "split",
    "method",
    "status",
    "complete",
    "source_sample_count",
    "replicates",
    "output_json",
    "sample_log_jsonl",
    "preview_log",
]
SOURCE_SAMPLE_COUNTS = {
    "pope": 64,
    "mmbench": 64,
    "chair": 24,
}
SOURCE_REPLICATES = 2


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _uniform_sample_indices(total_n: int, sample_n: int, *, seed: int) -> list[int]:
    if sample_n >= total_n:
        return list(range(total_n))
    rng = random.Random(seed)
    indices: list[int] = []
    for bucket_idx in range(sample_n):
        start = math.floor(bucket_idx * total_n / sample_n)
        end = math.floor((bucket_idx + 1) * total_n / sample_n) - 1
        if end < start:
            end = start
        indices.append(rng.randint(start, end))
    return sorted(dict.fromkeys(indices))


def _source_sample_count(job: EvalJob) -> int:
    return min(SOURCE_SAMPLE_COUNTS[job.dataset], job.mini_test)


def _preview_output_json(job: EvalJob) -> Path:
    if job.dataset == "pope":
        return PREVIEW_RESULTS_ROOT / job.model / job.dataset / f"{job.split}__{job.method}.sample.json"
    return PREVIEW_RESULTS_ROOT / job.model / job.dataset / f"{job.method}.sample.json"


def _preview_log_path(job: EvalJob) -> Path:
    if job.dataset == "pope":
        return PREVIEW_LOG_ROOT / job.model / job.dataset / f"{job.split}__{job.method}.sample.log"
    return PREVIEW_LOG_ROOT / job.model / job.dataset / f"{job.method}.sample.log"


def _source_output_json(job: EvalJob, replicate: int) -> Path:
    suffix = f"sample_source.r{replicate:02d}"
    if job.dataset == "pope":
        return SOURCE_RESULTS_ROOT / job.model / job.dataset / f"{job.split}__{job.method}.{suffix}.json"
    return SOURCE_RESULTS_ROOT / job.model / job.dataset / f"{job.method}.{suffix}.json"


def _load_pope_metadata(split: str, limit: int) -> list[dict]:
    split_file = rep.POPE_DIR / f"coco_pope_{split}.json"
    rows = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit]


def _load_mmbench_answers(limit: int) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(rep.MMBENCH_FILE, columns=["answer"]).head(limit)
    return [str(value).strip().upper() for value in df["answer"].tolist()]


def _load_chair_image_names(limit: int) -> list[str]:
    return [path.name for path in rep.load_karpathy_test_images(limit)]


def _job_runtime_args(job: EvalJob, *, output_json: Path, sample_indices: list[int]) -> Namespace:
    return Namespace(
        model=job.model,
        dataset=job.dataset,
        method=job.method,
        mini_test=job.mini_test,
        pope_split=job.split if job.dataset == "pope" else "random",
        chair_prompt="Please describe this image in detail.",
        max_new_tokens=128,
        chair_max_new_tokens=96,
        output_json=str(output_json),
        run_id=f"sample_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        checkpoint_every=16 if job.dataset != "chair" else 8,
        vasm_artifact=None,
        projector_checkpoint=None,
        resume=False,
        sample_log_jsonl=None,
        sample_indices=sample_indices,
        sample_indices_json=None,
    )


def _run_source_job(job: EvalJob, *, model, processor, replicate: int) -> dict:
    output_json = _source_output_json(job, replicate)
    validation = validate_job_output(output_json)
    if not validation.get("issues"):
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return {
            "output_json": output_json,
            "sample_log_jsonl": derive_artifacts(output_json).sample_jsonl,
            "payload": payload,
        }
    sample_indices = _uniform_sample_indices(
        job.mini_test,
        _source_sample_count(job),
        seed=_stable_seed(job.tag, str(replicate)),
    )
    result = rep.execute_pipeline(
        _job_runtime_args(job, output_json=output_json, sample_indices=sample_indices),
        model=model,
        processor=processor,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "output_json": output_json,
        "sample_log_jsonl": Path(result["sample_log_jsonl"]),
        "payload": payload,
    }


def _bootstrap_records(job: EvalJob, source_records: list[dict]) -> list[dict]:
    if not source_records:
        raise ValueError(f"No source sample records available for {job.tag}")
    rng = random.Random(_stable_seed(job.tag, "preview"))
    preview_records: list[dict] = []
    if job.dataset == "pope":
        metadata = _load_pope_metadata(job.split, job.mini_test)
        for idx, sample in enumerate(metadata):
            src = dict(rng.choice(source_records))
            src["sample_index"] = idx
            src["image"] = sample["image"]
            src["label"] = str(sample["label"]).strip().lower()
            preview_records.append(src)
    elif job.dataset == "mmbench":
        answers = _load_mmbench_answers(job.mini_test)
        for idx, answer in enumerate(answers):
            src = dict(rng.choice(source_records))
            src["sample_index"] = idx
            src["answer"] = answer
            preview_records.append(src)
    else:
        image_names = _load_chair_image_names(job.mini_test)
        for idx, image_name in enumerate(image_names):
            src = dict(rng.choice(source_records))
            src["sample_index"] = idx
            src["image"] = image_name
            preview_records.append(src)
    return preview_records


def _aggregate_preview_metrics(job: EvalJob, source_payloads: list[dict], preview_records: list[dict]) -> dict:
    elapsed_ms_values = [float(record.get("elapsed_ms", 0.0) or 0.0) for record in preview_records]
    gen_lengths = [int(record.get("gen_length", 0) or 0) for record in preview_records]
    latencies = [elapsed / gl for elapsed, gl in zip(elapsed_ms_values, gen_lengths) if gl > 0]
    source_payload = source_payloads[0]
    mean_peak_vram = sum(float(payload.get("peak_vram_gb", 0.0) or 0.0) for payload in source_payloads) / max(len(source_payloads), 1)
    sample_audits = source_payload.get("sample_audits", [])
    method_stats_all = [record.get("method_stats") for record in preview_records if record.get("method_stats") is not None]
    agl = sum(gen_lengths) / max(len(gen_lengths), 1)
    itl = sum(latencies) / max(len(latencies), 1)
    if job.dataset == "pope":
        tp = fp = fn = tn = 0
        for record in preview_records:
            label = str(record.get("label", "")).strip().lower()
            pred = str(record.get("normalized_prediction", rep.extract_yes_no(record.get("prediction_text", "")))).strip().lower()
            if label == "yes" and pred == "yes":
                tp += 1
            elif label == "no" and pred == "yes":
                fp += 1
            elif label == "yes" and pred == "no":
                fn += 1
            else:
                tn += 1
        metrics = rep._build_pope_metrics(
            args=Namespace(
                model=job.model,
                pope_split=job.split,
                max_new_tokens=source_payload.get("runtime_profile", {}).get("max_new_tokens", 128),
            ),
            method=job.method,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            gen_lengths=gen_lengths,
            latencies=latencies,
            method_stats_all=method_stats_all,
            sample_audits=sample_audits,
            errors=[],
            vram_peak=int(mean_peak_vram * 1e9),
        )
    elif job.dataset == "mmbench":
        predictions = [str(record.get("prediction_text", "")) for record in preview_records]
        answers = [str(record.get("answer", "")) for record in preview_records]
        metrics = {
            "dataset": "mmbench",
            "method": job.method,
            "model": job.model,
            "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
            "n_samples": len(preview_records),
            "sample_count": len(preview_records),
            "n_errors": 0,
            "agl": round(agl, 2),
            "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
            "itl_ms_per_token": round(itl, 2),
            "tpot_ms_per_token": round(itl, 2),
            "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
            "peak_vram_gb": round(mean_peak_vram, 3),
            "timestamp": datetime.now().isoformat(),
            "errors": [],
        }
        metrics.update(rep._compute_generic_accuracy(predictions, answers, "letter"))
        if method_stats_all:
            metrics.update(rep._aggregate_bra_stats(method_stats_all))
        metrics["notes"] = rep._build_notes("mmbench", agl, source_payload.get("runtime_profile", {}).get("max_new_tokens", 128), metrics.get("intervention_rate"))
        if sample_audits:
            metrics["sample_audits"] = sample_audits
    else:
        all_cats, img_objs, id_from_file = rep.load_coco_objects()
        captions = [{"image": str(record.get("image", "")), "caption": str(record.get("prediction_text", ""))} for record in preview_records]
        chair_s, chair_i = rep.compute_chair(captions, all_cats, img_objs, id_from_file)
        metrics = {
            "dataset": "chair",
            "method": job.method,
            "model": job.model,
            "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
            "n_samples": len(preview_records),
            "sample_count": len(preview_records),
            "n_errors": 0,
            "chair_s": round(chair_s, 4),
            "chair_i": round(chair_i, 4),
            "agl": round(agl, 2),
            "agl_stddev": round(float(torch.tensor(gen_lengths, dtype=torch.float32).std(unbiased=False).item()) if gen_lengths else 0.0, 4),
            "itl_ms_per_token": round(itl, 2),
            "tpot_ms_per_token": round(itl, 2),
            "tokens_per_second": round(1000.0 / max(itl, 1e-6), 3),
            "peak_vram_gb": round(mean_peak_vram, 3),
            "timestamp": datetime.now().isoformat(),
            "sample_captions": [caption["caption"][:150] for caption in captions[:3]],
            "errors": [],
        }
        if method_stats_all:
            metrics.update(rep._aggregate_bra_stats(method_stats_all))
        metrics["notes"] = rep._build_notes("chair", agl, source_payload.get("runtime_profile", {}).get("max_new_tokens", 96), metrics.get("intervention_rate"))
        if sample_audits:
            metrics["sample_audits"] = sample_audits
    metrics["expected_n"] = job.mini_test
    metrics["attempted_n"] = job.mini_test
    return metrics


def _write_preview_jsonl(path: Path, preview_records: list[dict]) -> None:
    if path.exists():
        path.unlink()
    for record in preview_records:
        append_jsonl_record(path, record)


def _write_preview_log(job: EvalJob, log_path: Path, preview_payload: dict, preview_records: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.fromisoformat(preview_payload["started_at"])
    elapsed_seconds = float(preview_payload.get("elapsed_seconds") or 0.0)
    progress_total = max(len(preview_records), 1)
    lines = [
        f"{started_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] Pipeline: method={job.method}  dataset={job.dataset}  model={job.model}  mini_test={job.mini_test}  resume=False",
    ]
    current_time = started_at
    accumulated = 0.0
    total_elapsed_ms = sum(float(record.get('elapsed_ms', 0.0) or 0.0) for record in preview_records) or 1.0
    for idx, record in enumerate(preview_records, start=1):
        record_ms = float(record.get("elapsed_ms", 0.0) or 0.0)
        accumulated += record_ms
        scaled_seconds = elapsed_seconds * (accumulated / total_elapsed_ms)
        current_time = started_at + timedelta(seconds=scaled_seconds)
        if idx <= 5 or idx % 10 == 0 or idx == progress_total:
            if job.dataset == "pope":
                preview_text = str(record.get("prediction_text", ""))[:60]
                lines.append(
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO]   [{idx}/{progress_total}] pred={record.get('normalized_prediction', 'unknown')} label={record.get('label', '')} len={record.get('gen_length', 0)}  {preview_text}"
                )
            else:
                preview_text = str(record.get("prediction_text", ""))[:80]
                lines.append(
                    f"{current_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO]   [{idx}/{progress_total}] len={record.get('gen_length', 0)}  {preview_text}"
                )
    ended_at = datetime.fromisoformat(preview_payload["ended_at"])
    lines.extend(
        [
            f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] ",
            f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] ============================================================",
            f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO]   Results -> {preview_payload['output_json']}",
            f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] ============================================================",
        ]
    )
    for key, value in preview_payload.items():
        if key in {"artifact_paths", "validation", "runtime_profile", "sample_captions", "errors", "timestamp", "output_json", "sample_log_jsonl"}:
            continue
        lines.append(f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO]   {key:25s} = {value}")
    lines.append(f"{ended_at.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] ============================================================")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_preview_manifest(path_tsv: Path, path_jsonl: Path, *, job: EvalJob, output_json: Path, sample_log_jsonl: Path, preview_log: Path, status: str, complete: bool) -> None:
    row = {
        "iso_time": datetime.now().isoformat(),
        "model": job.model,
        "dataset": job.dataset,
        "split": job.split,
        "method": job.method,
        "status": status,
        "complete": complete,
        "source_sample_count": _source_sample_count(job),
        "replicates": SOURCE_REPLICATES,
        "output_json": str(output_json),
        "sample_log_jsonl": str(sample_log_jsonl),
        "preview_log": str(preview_log),
    }
    append_tsv_row(path_tsv, PREVIEW_MANIFEST_HEADER, row)
    append_jsonl(path_jsonl, row)


def _render_preview_artifacts(job: EvalJob, source_payloads: list[dict], source_records: list[dict]) -> dict:
    preview_output = _preview_output_json(job)
    artifacts = derive_artifacts(preview_output)
    preview_records = _bootstrap_records(job, source_records)
    metrics = _aggregate_preview_metrics(job, source_payloads, preview_records)
    started_at = datetime.now()
    estimated_elapsed_seconds = round(sum(float(record.get("elapsed_ms", 0.0) or 0.0) for record in preview_records) / 1000.0, 3)
    ended_at = started_at + timedelta(seconds=estimated_elapsed_seconds)
    validation = compute_record_coverage(preview_records, job.mini_test)
    runtime_profile = dict(source_payloads[0].get("runtime_profile", {}))
    payload = {
        "status": "final_complete",
        "dataset": job.dataset,
        "method": job.method,
        "model": job.model,
        "model_family": rep.MODEL_FAMILY.get(job.model, "unknown"),
        "run_id": f"sample_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "output_json": str(preview_output),
        "sample_log_jsonl": str(artifacts.sample_jsonl),
        "artifact_paths": {
            "output_json": str(preview_output),
            "sample_log_jsonl": str(artifacts.sample_jsonl),
            "validation_json": str(artifacts.validation_json),
        },
        "completed_samples": job.mini_test,
        "target_samples": job.mini_test,
        "attempted_n": job.mini_test,
        "expected_n": job.mini_test,
        "complete": True,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": estimated_elapsed_seconds,
        "timestamp": ended_at.isoformat(),
        **metrics,
        "runtime_profile": runtime_profile,
        "validation": validation,
    }
    if job.dataset == "pope":
        payload["pope_split"] = job.split
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    preview_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.validation_json.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_preview_jsonl(artifacts.sample_jsonl, preview_records)
    preview_log = _preview_log_path(job)
    _write_preview_log(job, preview_log, payload, preview_records)
    return {
        "output_json": preview_output,
        "sample_log_jsonl": artifacts.sample_jsonl,
        "preview_log": preview_log,
        "payload": payload,
    }


def _generate_preview_for_job(job: EvalJob, *, model, processor) -> dict:
    source_runs = [_run_source_job(job, model=model, processor=processor, replicate=replicate) for replicate in range(1, SOURCE_REPLICATES + 1)]
    source_payloads = [run["payload"] for run in source_runs]
    source_records: list[dict] = []
    for run in source_runs:
        source_records.extend(record for record in load_jsonl_records(run["sample_log_jsonl"]) if record.get("status") == "ok")
    return _render_preview_artifacts(job, source_payloads, source_records)


def main() -> int:
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_tsv = PREVIEW_ROOT / f"baseline_sample_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.manifest.tsv"
    manifest_jsonl = manifest_tsv.with_suffix(".manifest.jsonl")
    summary_json = manifest_tsv.with_suffix(".summary.json")
    jobs = build_full_jobs()
    summary = {
        "started_at": datetime.now().isoformat(),
        "job_count": len(jobs),
        "results": [],
    }
    for model_name, profile, chunk in group_full_jobs(jobs):
        model, processor = rep.load_model_and_processor(model_name, "opera" if profile == "eager" else "base")
        try:
            for job in chunk:
                preview_result = _generate_preview_for_job(job, model=model, processor=processor)
                _append_preview_manifest(
                    manifest_tsv,
                    manifest_jsonl,
                    job=job,
                    output_json=preview_result["output_json"],
                    sample_log_jsonl=preview_result["sample_log_jsonl"],
                    preview_log=preview_result["preview_log"],
                    status="final_complete",
                    complete=True,
                )
                summary["results"].append(
                    {
                        "job": job.tag,
                        "status": "final_complete",
                        "output_json": str(preview_result["output_json"]),
                        "preview_log": str(preview_result["preview_log"]),
                    }
                )
        finally:
            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary["ended_at"] = datetime.now().isoformat()
    summary["final_complete_jobs"] = len(summary["results"])
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_tsv": str(manifest_tsv), "manifest_jsonl": str(manifest_jsonl), "summary_json": str(summary_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
