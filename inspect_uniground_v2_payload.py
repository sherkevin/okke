#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate a UniGround v2 feature payload.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def _tensor_stats(tensor: torch.Tensor) -> dict:
    tensor = tensor.float()
    norms = tensor.norm(dim=-1) if tensor.ndim >= 2 else tensor.abs()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "mean_norm": round(float(norms.mean().item()), 6),
        "std_norm": round(float(norms.std(unbiased=False).item()), 6),
        "min_norm": round(float(norms.min().item()), 6),
        "max_norm": round(float(norms.max().item()), 6),
    }


def inspect_payload(payload: dict) -> dict:
    image_embeddings = payload["image_embeddings"].float()
    candidate_embeddings = payload["candidate_embeddings"].float()
    prefix_embeddings = payload["prefix_embeddings"].float()
    region_embeddings = payload.get("retrieved_region_embeddings", payload.get("region_embeddings")).float()
    labels = payload["labels"].float()
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}

    label_counts = {
        "support": int(labels[:, 0].sum().item()) if labels.ndim == 2 else None,
        "contradiction": int(labels[:, 1].sum().item()) if labels.ndim == 2 else None,
        "abstain": int(labels[:, 2].sum().item()) if labels.ndim == 2 else None,
    }
    region_norms = region_embeddings.norm(dim=-1)
    zero_region_rows = int((region_norms.max(dim=-1).values <= 1e-8).sum().item()) if region_embeddings.ndim == 3 else 0

    checks = {
        "record_alignment": (
            image_embeddings.shape[0]
            == candidate_embeddings.shape[0]
            == prefix_embeddings.shape[0]
            == region_embeddings.shape[0]
            == labels.shape[0]
        ),
        "embed_dim_alignment": (
            image_embeddings.shape[-1]
            == candidate_embeddings.shape[-1]
            == prefix_embeddings.shape[-1]
            == region_embeddings.shape[-1]
        ),
        "regions_present": bool(region_embeddings.ndim == 3 and region_embeddings.shape[1] >= 1),
        "all_finite": bool(
            torch.isfinite(image_embeddings).all().item()
            and torch.isfinite(candidate_embeddings).all().item()
            and torch.isfinite(prefix_embeddings).all().item()
            and torch.isfinite(region_embeddings).all().item()
            and torch.isfinite(labels).all().item()
        ),
        "normalized_embeddings": all(
            0.80 <= stats["mean_norm"] <= 1.20
            for stats in (
                _tensor_stats(image_embeddings),
                _tensor_stats(candidate_embeddings),
                _tensor_stats(prefix_embeddings),
            )
        ),
        "balanced_labels": (
            max(label_counts.values()) - min(label_counts.values()) <= max(1, int(0.05 * labels.shape[0]))
            if all(value is not None for value in label_counts.values())
            else False
        ),
        "nonzero_regions": zero_region_rows == 0,
        "metadata_record_count_match": int(metadata.get("record_count", -1)) == int(labels.shape[0]),
        "metadata_no_llm_labels": bool(not metadata.get("augmentation_policy", {}).get("llm_used", True)),
    }
    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks,
        "record_count": int(labels.shape[0]),
        "image_count": int(metadata.get("image_count", -1)),
        "candidate_vocab_size": int(metadata.get("candidate_vocab_size", -1)),
        "prefix_count": int(metadata.get("prefix_count", -1)),
        "label_counts": label_counts,
        "metadata_label_counts": metadata.get("label_counts"),
        "caption_coverage_rate": metadata.get("caption_coverage_rate"),
        "image_embeddings": _tensor_stats(image_embeddings),
        "candidate_embeddings": _tensor_stats(candidate_embeddings),
        "prefix_embeddings": _tensor_stats(prefix_embeddings),
        "region_embeddings": _tensor_stats(region_embeddings.reshape(-1, region_embeddings.shape[-1])),
        "region_shape": list(region_embeddings.shape),
        "zero_region_rows": zero_region_rows,
        "augmentation_policy": metadata.get("augmentation_policy"),
        "region_mode": metadata.get("region_mode"),
    }


def main() -> None:
    args = parse_args()
    payload_path = Path(args.payload)
    payload = torch.load(payload_path, map_location="cpu")
    report = inspect_payload(payload)
    report["payload_path"] = str(payload_path.resolve())
    report["payload_size_bytes"] = payload_path.stat().st_size
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
