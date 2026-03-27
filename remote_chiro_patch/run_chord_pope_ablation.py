from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ALPHA_ANCHOR = 0.5
DEFAULT_LAMBDA_CUR = 0.25
DEFAULT_LAMBDA_FUT = 0.5
DEFAULT_LAMBDA_TXT = 1.0
DEFAULT_FUTURE_HORIZON = 4
DEFAULT_FUTURE_TOPK = 5
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_MAX_BOXES = 10

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}

RUNNER_SCRIPT = r"""
import json
import sys
from pathlib import Path

import pope_eval

payload = json.loads(sys.argv[1])
pope_key = payload["pope_key"]
pope_eval.POPE_PATH[pope_key] = payload["pope_path"]
sys.argv = ["pope_eval.py"] + payload["argv"]
pope_eval.main()
"""


@dataclass(frozen=True)
class AblationSpec:
    name: str
    chord_enable: bool
    alpha_anchor: float = DEFAULT_ALPHA_ANCHOR
    lambda_cur: float = DEFAULT_LAMBDA_CUR
    lambda_fut: float = DEFAULT_LAMBDA_FUT
    lambda_txt: float = DEFAULT_LAMBDA_TXT
    future_horizon: int = DEFAULT_FUTURE_HORIZON
    future_topk: int = DEFAULT_FUTURE_TOPK
    detector_box_threshold: float = DEFAULT_BOX_THRESHOLD
    detector_text_threshold: float = DEFAULT_TEXT_THRESHOLD
    max_boxes: int = DEFAULT_MAX_BOXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-slice CHORD ablations on the official POPE path.")
    parser.add_argument("--split", choices=sorted(POPE_PATH), default="random")
    parser.add_argument("--limit", type=int, default=128, help="Number of POPE samples to keep in the fixed slice.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset inside the original split.")
    parser.add_argument("--output-dir", type=str, default="tests/ablations")
    parser.add_argument("--suite", choices=["core", "spec"], default="spec")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", type=str, default="0,1")
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--scale-factor", type=float, default=50.0)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num-attn-candidates", type=int, default=5)
    parser.add_argument("--penalty-weights", type=float, default=1.0)
    parser.add_argument("--detector-device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--alpha-grid", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    parser.add_argument("--lambda-cur-grid", type=float, nargs="*", default=[])
    parser.add_argument("--lambda-fut-grid", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    parser.add_argument("--horizon-grid", type=int, nargs="*", default=[2, 4, 6])
    parser.add_argument("--future-topk-grid", type=int, nargs="*", default=[])
    return parser.parse_args()


def _fmt_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _build_fixed_slice(split: str, limit: int, offset: int, output_dir: Path) -> Path:
    source_path = Path(POPE_PATH[split])
    lines = source_path.read_text(encoding="utf-8").splitlines()
    selected = lines[offset : offset + limit]
    if not selected:
        raise ValueError(f"No POPE samples selected for split={split!r}, offset={offset}, limit={limit}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = output_dir / f"{split}_slice_{offset}_{limit}.json"
    slice_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return slice_path


def _base_full_spec(name: str = "chord_full_default", **overrides: float | int | bool | str) -> AblationSpec:
    kwargs = {
        "name": name,
        "chord_enable": True,
        "alpha_anchor": DEFAULT_ALPHA_ANCHOR,
        "lambda_cur": DEFAULT_LAMBDA_CUR,
        "lambda_fut": DEFAULT_LAMBDA_FUT,
        "lambda_txt": DEFAULT_LAMBDA_TXT,
        "future_horizon": DEFAULT_FUTURE_HORIZON,
        "future_topk": DEFAULT_FUTURE_TOPK,
        "detector_box_threshold": DEFAULT_BOX_THRESHOLD,
        "detector_text_threshold": DEFAULT_TEXT_THRESHOLD,
        "max_boxes": DEFAULT_MAX_BOXES,
    }
    kwargs.update(overrides)
    return AblationSpec(**kwargs)


def _build_specs(args: argparse.Namespace) -> list[AblationSpec]:
    specs: list[AblationSpec] = [
        AblationSpec(name="opera", chord_enable=False),
        _base_full_spec(name="anchor_current_only", lambda_fut=0.0),
        _base_full_spec(name="future_only", lambda_cur=0.0),
        _base_full_spec(name="chord_full_default"),
    ]

    if args.suite == "spec":
        specs.append(_base_full_spec(name="chord_full_no_gino", alpha_anchor=0.0))
        for horizon in args.horizon_grid:
            specs.append(_base_full_spec(name=f"chord_full_h{horizon}", future_horizon=int(horizon)))
        for alpha in args.alpha_grid:
            specs.append(_base_full_spec(name=f"chord_full_alpha_{_fmt_float(alpha)}", alpha_anchor=float(alpha)))
        for lambda_fut in args.lambda_fut_grid:
            specs.append(_base_full_spec(name=f"chord_full_lfut_{_fmt_float(lambda_fut)}", lambda_fut=float(lambda_fut)))

    for lambda_cur in args.lambda_cur_grid:
        specs.append(_base_full_spec(name=f"chord_full_lcur_{_fmt_float(lambda_cur)}", lambda_cur=float(lambda_cur)))
    for future_topk in args.future_topk_grid:
        specs.append(_base_full_spec(name=f"chord_full_ftopk_{future_topk}", future_topk=int(future_topk)))

    deduped: list[AblationSpec] = []
    seen: set[tuple] = set()
    for spec in specs:
        spec_key = tuple(asdict(spec).items())
        if spec_key in seen:
            continue
        seen.add(spec_key)
        deduped.append(spec)
    return deduped


def _build_run_argv(args: argparse.Namespace, spec: AblationSpec, output_json: Path, log_jsonl: Path, pope_key: str) -> list[str]:
    argv = [
        "--model",
        args.model,
        "--pope-type",
        pope_key,
        "--gpu-id",
        str(args.gpu_id),
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--data_path",
        args.data_path,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--beam",
        str(args.beam),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--scale_factor",
        str(args.scale_factor),
        "--threshold",
        str(args.threshold),
        "--num_attn_candidates",
        str(args.num_attn_candidates),
        "--penalty_weights",
        str(args.penalty_weights),
        "--output-json",
        str(output_json),
        "--chord-log-jsonl",
        str(log_jsonl),
    ]
    if spec.chord_enable:
        argv.extend(
            [
                "--chord-enable",
                "--alpha-anchor",
                str(spec.alpha_anchor),
                "--lambda-cur",
                str(spec.lambda_cur),
                "--lambda-fut",
                str(spec.lambda_fut),
                "--lambda-txt",
                str(spec.lambda_txt),
                "--future-horizon",
                str(spec.future_horizon),
                "--future-topk",
                str(spec.future_topk),
                "--detector-box-threshold",
                str(spec.detector_box_threshold),
                "--detector-text-threshold",
                str(spec.detector_text_threshold),
                "--max-boxes",
                str(spec.max_boxes),
                "--detector-device",
                args.detector_device,
            ]
        )
    return argv


def _run_one_spec(
    *,
    repo_root: Path,
    python_bin: str,
    env: dict[str, str],
    args: argparse.Namespace,
    spec: AblationSpec,
    slice_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, float]:
    output_json = output_dir / f"{spec.name}.json"
    log_jsonl = output_dir / f"{spec.name}.jsonl"
    pope_key = f"ablation_{args.split}_{args.offset}_{args.limit}"
    payload = {
        "pope_key": pope_key,
        "pope_path": str(slice_path),
        "argv": _build_run_argv(args, spec, output_json, log_jsonl, pope_key),
    }
    start = time.time()
    proc = subprocess.run(
        [python_bin, "-c", RUNNER_SCRIPT, json.dumps(payload, ensure_ascii=False)],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed_s = time.time() - start
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Ablation run {spec.name!r} failed with exit code {proc.returncode}.")
    return output_json, log_jsonl, elapsed_s


def _load_summary(output_path: Path) -> dict:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs", [])
    predictions: list[int] = []
    labels: list[int] = []
    future_active_count = 0
    future_failed_count = 0
    rollback_count = 0

    for record in outputs:
        label = int(record["label"][0])
        answer = str(record["answer"][0]).strip().lower()
        pred = 1 if answer.startswith("yes") else 0
        labels.append(label)
        predictions.append(pred)

        steps = [entry for entry in record.get("decode_diagnostics", []) if isinstance(entry, dict) and "future_failed" in entry]
        if not steps:
            continue
        step = steps[-1]
        if step.get("rollback_triggered"):
            rollback_count += 1
        future_failed = [flag for row in step.get("future_failed", []) for flag in row]
        future_scores = [float(value) for row in step.get("f_future", []) for value in row]
        if any(future_failed):
            future_failed_count += 1
        if any(abs(value) > 1e-8 for value in future_scores):
            future_active_count += 1

    return {
        "metrics": payload.get("metrics", {}),
        "predictions": predictions,
        "labels": labels,
        "future_active_count": future_active_count,
        "future_failed_count": future_failed_count,
        "rollback_count": rollback_count,
        "output_path": str(output_path),
    }


def _compare_to_reference(reference: dict, candidate: dict) -> dict:
    ref_preds = reference["predictions"]
    cur_preds = candidate["predictions"]
    labels = reference["labels"]
    if len(ref_preds) != len(cur_preds) or len(ref_preds) != len(labels):
        raise ValueError("Reference and candidate runs do not share the same fixed slice length.")

    improved = 0
    worsened = 0
    changed = 0
    for ref_pred, cur_pred, label in zip(ref_preds, cur_preds, labels):
        if ref_pred != cur_pred:
            changed += 1
            if ref_pred != label and cur_pred == label:
                improved += 1
            elif ref_pred == label and cur_pred != label:
                worsened += 1

    ref_metrics = reference["metrics"]
    cur_metrics = candidate["metrics"]
    return {
        "changed_vs_opera": changed,
        "improved_vs_opera": improved,
        "worsened_vs_opera": worsened,
        "delta_accuracy_vs_opera": float(cur_metrics.get("accuracy", 0.0)) - float(ref_metrics.get("accuracy", 0.0)),
        "delta_f1_vs_opera": float(cur_metrics.get("f1", 0.0)) - float(ref_metrics.get("f1", 0.0)),
    }


def _write_markdown(summary_path: Path, report: dict) -> None:
    lines = [
        f"# CHORD POPE Ablation Summary: {report['split']}",
        "",
        f"- Slice: `{report['slice_path']}`",
        f"- Limit: `{report['limit']}`",
        f"- Offset: `{report['offset']}`",
        "",
        "| Spec | Acc | F1 | Future Active | Future Failed | Changed vs OPERA | Improved | Worsened |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["runs"]:
        metrics = row["metrics"]
        comparison = row.get("comparison_to_opera", {})
        lines.append(
            "| {name} | {acc:.4f} | {f1:.4f} | {active} | {failed} | {changed} | {improved} | {worsened} |".format(
                name=row["name"],
                acc=float(metrics.get("accuracy", 0.0)),
                f1=float(metrics.get("f1", 0.0)),
                active=row["future_active_count"],
                failed=row["future_failed_count"],
                changed=comparison.get("changed_vs_opera", 0),
                improved=comparison.get("improved_vs_opera", 0),
                worsened=comparison.get("worsened_vs_opera", 0),
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir / f"{args.split}_{args.offset}_{args.limit}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_path = _build_fixed_slice(args.split, args.limit, args.offset, output_dir)
    specs = _build_specs(args)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'transformers-4.29.2/src'}"

    runs: list[dict] = []
    opera_summary: dict | None = None

    for spec in specs:
        print(f"=== RUN {spec.name} ===", flush=True)
        output_json, log_jsonl, elapsed_s = _run_one_spec(
            repo_root=repo_root,
            python_bin=args.python_bin,
            env=env,
            args=args,
            spec=spec,
            slice_path=slice_path,
            output_dir=output_dir,
        )
        summary = _load_summary(output_json)
        run_record = {
            "name": spec.name,
            "spec": asdict(spec),
            "metrics": summary["metrics"],
            "future_active_count": summary["future_active_count"],
            "future_failed_count": summary["future_failed_count"],
            "rollback_count": summary["rollback_count"],
            "elapsed_s": elapsed_s,
            "output_json": str(output_json),
            "log_jsonl": str(log_jsonl),
        }
        if opera_summary is None:
            opera_summary = summary
        else:
            run_record["comparison_to_opera"] = _compare_to_reference(opera_summary, summary)
        runs.append(run_record)
        print(json.dumps(run_record, ensure_ascii=False, indent=2))

    report = {
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "slice_path": str(slice_path),
        "runs": runs,
    }
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    print(f"Wrote ablation summary to {json_path}")
    print(f"Wrote ablation markdown to {md_path}")


if __name__ == "__main__":
    main()
