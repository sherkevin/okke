#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


NEG_WORDS = {"No", "not", "no", "NO"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def predict_yes(answer: Any) -> int:
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    text = str(answer).replace(".", "").replace(",", "")
    words = text.split(" ")
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return 0
    return 1


def labels_and_predictions(payload: dict[str, Any]) -> tuple[list[int], list[int]]:
    labels: list[int] = []
    preds: list[int] = []
    for item in payload.get("outputs", []):
        label = item.get("label", [None])
        if isinstance(label, list):
            label = label[0] if label else None
        labels.append(int(label))
        preds.append(predict_yes(item.get("answer", "")))
    return labels, preds


def safe_get(payload: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fp_rate(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    metrics = payload.get("metrics", {})
    fp = metrics.get("fp")
    tn = metrics.get("tn")
    if fp is None or tn is None or (fp + tn) == 0:
        return None
    return fp / (fp + tn)


def variant_row(name: str, payload: dict[str, Any] | None, chair_s: float | None = None) -> dict[str, Any]:
    metrics = payload.get("metrics", {}) if payload else {}
    summary = payload.get("diagnostics_summary", {}) if payload else {}
    return {
        "name": name,
        "json": str(payload.get("_path")) if payload else None,
        "n": len(payload.get("outputs", [])) if payload else 0,
        "adv_f1": metrics.get("f1"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "fp_rate": fp_rate(payload),
        "chair_s": chair_s,
        "mean_sample_time_ms": summary.get("mean_sample_time_ms"),
        "mean_generated_tokens": summary.get("mean_generated_tokens"),
        "peak_vram_gb": payload.get("peak_vram_gb") if payload else None,
        "present": payload is not None,
    }


def paired_flip_row(name: str, pc: dict[str, Any] | None, full: dict[str, Any] | None) -> dict[str, Any]:
    if pc is None or full is None:
        return {
            "model_task": name,
            "contrast": "Full-P+C",
            "present": False,
            "evidence_boundary": "missing paired P+C or Full JSON",
        }
    pc_labels, pc_preds = labels_and_predictions(pc)
    full_labels, full_preds = labels_and_predictions(full)
    n = min(len(pc_labels), len(pc_preds), len(full_labels), len(full_preds))
    corrected = 0
    harmful = 0
    flips = 0
    for idx in range(n):
        pc_ok = pc_preds[idx] == pc_labels[idx]
        full_ok = full_preds[idx] == full_labels[idx]
        if pc_preds[idx] != full_preds[idx]:
            flips += 1
            if not pc_ok and full_ok:
                corrected += 1
            elif pc_ok and not full_ok:
                harmful += 1
    pc_f1 = safe_get(pc, "metrics", "f1")
    full_f1 = safe_get(full, "metrics", "f1")
    return {
        "model_task": name,
        "contrast": "Full-P+C",
        "present": True,
        "n": n,
        "flip_rate": flips / n if n else None,
        "flips": flips,
        "corrected": corrected,
        "harmful": harmful,
        "metric": "POPE adversarial F1",
        "metric_delta": (full_f1 - pc_f1) if full_f1 is not None and pc_f1 is not None else None,
        "pc_json": str(pc.get("_path")),
        "full_json": str(full.get("_path")),
    }


def efficiency_row(name: str, payload: dict[str, Any] | None, proposal_ms: float | None = None) -> dict[str, Any]:
    if payload is None:
        return {"regime": name, "present": False}
    summary = payload.get("diagnostics_summary", {})
    decode_ms = summary.get("mean_sample_time_ms")
    tokens = summary.get("mean_generated_tokens")
    total_ms = None
    if decode_ms is not None:
        total_ms = float(decode_ms) + float(proposal_ms or 0.0)
    itl = None
    if decode_ms is not None and tokens:
        itl = float(decode_ms) / float(tokens)
    return {
        "regime": name,
        "present": True,
        "proposal_ms": proposal_ms,
        "decode_itl_ms_per_token": itl,
        "mean_generated_tokens": tokens,
        "total_ms_per_sample": total_ms,
        "peak_vram_gb": payload.get("peak_vram_gb"),
        "n": len(payload.get("outputs", [])),
        "json": str(payload.get("_path")),
    }


def load_named(run_dir: Path, names: dict[str, str]) -> dict[str, dict[str, Any] | None]:
    loaded: dict[str, dict[str, Any] | None] = {}
    for key, rel in names.items():
        path = run_dir / rel
        payload = load_json(path)
        if payload is not None:
            payload["_path"] = path
        loaded[key] = payload
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--tag", default="adv64")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    prefix = f"llava_pope_{args.tag}"
    names = {
        "opera": f"{prefix}_opera.json",
        "same_anchor_non_chord": f"{prefix}_same_anchor_non_chord.json",
        "pc_uniform": f"{prefix}_pc_uniform.json",
        "pc_random": f"{prefix}_pc_random.json",
        "past_future": f"{prefix}_past_future.json",
        "pc_real": f"{prefix}_pc_real.json",
        "full_k5_m3": f"{prefix}_full_k5_m3.json",
        "full_k3_m2": f"{prefix}_full_k3_m2.json",
        "full_k5_m1": f"{prefix}_full_k5_m1.json",
        "full_k5_m2": f"{prefix}_full_k5_m2.json",
        "full_k5_m4": f"{prefix}_full_k5_m4.json",
        "full_k10_m3": f"{prefix}_full_k10_m3.json",
    }
    data = load_named(run_dir, names)

    anchor_summary = load_json(run_dir / f"anchors_pope_{args.tag}_summary.json") or {}
    proposal_ms = anchor_summary.get("mean_detector_ms")

    detector_rows = [
        variant_row("OPERA / Past", data["opera"]),
        variant_row("Same-anchor non-CHORD", data["same_anchor_non_chord"]),
        variant_row("P+C with uniform anchors", data["pc_uniform"]),
        variant_row("P+C with random matched anchors", data["pc_random"]),
        variant_row("Past+Future, no Current", data["past_future"]),
        variant_row("P+C real anchors", data["pc_real"]),
        variant_row("Full real anchors", data["full_k5_m3"]),
    ]

    km_rows = [
        {"k": None, "m": 0, **variant_row("P+C reference", data["pc_real"])},
        {"k": 3, "m": 2, **variant_row("k3 m2", data["full_k3_m2"])},
        {"k": 5, "m": 1, **variant_row("k5 m1", data["full_k5_m1"])},
        {"k": 5, "m": 2, **variant_row("k5 m2", data["full_k5_m2"])},
        {"k": 5, "m": 3, **variant_row("k5 m3", data["full_k5_m3"])},
        {"k": 5, "m": 4, **variant_row("k5 m4", data["full_k5_m4"])},
        {"k": 10, "m": 3, **variant_row("k10 m3", data["full_k10_m3"])},
    ]

    payload = {
        "scope": "CHORD rebuttal real metrics for author_response_min_diff_expected_20260529.tex",
        "run_dir": str(run_dir),
        "tag": args.tag,
        "important_boundary": (
            "These are measured JSON/log values for the configured split. Rows with null values "
            "were not measured and must not be inserted into the PDF as real evidence."
        ),
        "table_1_future_mechanism": [
            paired_flip_row("LLaVA POPE-Adv", data["pc_real"], data["full_k5_m3"]),
            {
                "model_task": "InstructBLIP POPE-Adv",
                "present": False,
                "evidence_boundary": "not run in this LLaVA-focused remote pass",
            },
            {
                "model_task": "LLaVA CHAIR",
                "present": False,
                "evidence_boundary": "CHAIR captioning was not measured by pope_eval JSON",
            },
            {
                "model_task": "InstructBLIP CHAIR",
                "present": False,
                "evidence_boundary": "not run in this LLaVA-focused remote pass",
            },
        ],
        "table_2_detector_attribution": detector_rows,
        "table_3_efficiency": [
            efficiency_row("OPERA / Past", data["opera"], proposal_ms=0.0),
            efficiency_row("Past+Current", data["pc_real"], proposal_ms=proposal_ms),
            efficiency_row("Full CHORD", data["full_k5_m3"], proposal_ms=proposal_ms),
        ],
        "table_4_km_robustness": km_rows,
        "anchor_summary": anchor_summary,
        "present_json_count": sum(1 for value in data.values() if value is not None),
        "missing_json": [key for key, value in data.items() if value is None],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    write_json(out_json, payload)

    lines = [
        "# Real CHORD Rebuttal Metrics, 2026-05-29",
        "",
        f"Run dir: `{run_dir}`",
        "",
        f"Present JSON files: {payload['present_json_count']} / {len(names)}",
        "",
        "Boundary: rows with null values were not measured and must not be inserted into the PDF as real evidence.",
        "",
        "## Table 1 Future Mechanism",
        "",
    ]
    for row in payload["table_1_future_mechanism"]:
        lines.append(f"- {row.get('model_task')}: {json.dumps(row, ensure_ascii=False)}")
    lines.extend(["", "## Table 2 Detector Attribution", ""])
    for row in detector_rows:
        lines.append(f"- {row['name']}: F1={row.get('adv_f1')} FP={row.get('fp_rate')} CHAIR-S={row.get('chair_s')} n={row.get('n')}")
    lines.extend(["", "## Table 3 Efficiency", ""])
    for row in payload["table_3_efficiency"]:
        lines.append(f"- {row['regime']}: total_ms={row.get('total_ms_per_sample')} itl={row.get('decode_itl_ms_per_token')} vram={row.get('peak_vram_gb')} n={row.get('n')}")
    lines.extend(["", "## Table 4 k/m Robustness", ""])
    for row in km_rows:
        lines.append(f"- k={row.get('k')} m={row.get('m')}: F1={row.get('adv_f1')} total_ms={row.get('mean_sample_time_ms')} n={row.get('n')}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
