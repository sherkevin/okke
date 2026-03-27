#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from bra_universal_plugin import UniversalObservation, UniversalPluginOutput
from uniground_runtime import load_universal_scorer
from uniground_v2.scorer import HardcodedUniversalScorer
from uniground_v2.training_dataset import load_feature_payload, load_training_tensors_v2


CLASS_NAMES = ("support", "contradiction", "abstain")


@dataclass
class BatchResult:
    logits: torch.Tensor
    labels: torch.Tensor


class ZeroUniversalScorer:
    def __call__(
        self,
        observation: UniversalObservation,
        candidate_embeddings: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> UniversalPluginOutput:
        zeros = torch.zeros(candidate_embeddings.shape[0], device=candidate_embeddings.device, dtype=candidate_embeddings.dtype)
        return UniversalPluginOutput(support=zeros, contradiction=zeros.clone(), abstain=zeros.clone())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline module evaluation for Psi_univ.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--psi-checkpoint", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-size", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve_device(raw: str) -> torch.device:
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _build_eval_indices(total: int, eval_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(total, generator=generator)
    return perm[: min(total, eval_size)]


def _target_indices(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[-1] != 3:
        raise ValueError("Expected one-hot / multi-hot labels with shape [N, 3].")
    return labels.argmax(dim=-1).long()


def _macro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    f1s = []
    for cls in range(len(CLASS_NAMES)):
        tp = int(((pred == cls) & (target == cls)).sum().item())
        fp = int(((pred == cls) & (target != cls)).sum().item())
        fn = int(((pred != cls) & (target == cls)).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        denom = precision + recall
        f1s.append(0.0 if denom == 0 else (2.0 * precision * recall / denom))
    return float(sum(f1s) / len(f1s))


def _per_class_accuracy(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls, name in enumerate(CLASS_NAMES):
        mask = target == cls
        out[name] = float(((pred[mask] == target[mask]).float().mean().item()) if mask.any() else 0.0)
    return out


def evaluate_scorer(
    scorer,
    *,
    image_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    prefix_embeddings: torch.Tensor,
    region_embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> BatchResult:
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    total = image_embeddings.shape[0]

    if hasattr(scorer, "to"):
        scorer = scorer.to(device)
    if hasattr(scorer, "eval"):
        scorer.eval()

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        image_batch = image_embeddings[start:end].to(device)
        candidate_batch = candidate_embeddings[start:end].to(device)
        prefix_batch = prefix_embeddings[start:end].to(device)
        region_batch = region_embeddings[start:end].to(device)
        label_batch = labels[start:end].cpu()

        obs = UniversalObservation(image_embedding=image_batch, region_embeddings=region_batch)
        with torch.no_grad():
            output = scorer(obs, candidate_batch, prefix_batch)
            logits = torch.stack([output.support, output.contradiction, output.abstain], dim=-1).float().cpu()
        logits_parts.append(logits)
        label_parts.append(label_batch)

    return BatchResult(logits=torch.cat(logits_parts, dim=0), labels=torch.cat(label_parts, dim=0))


def summarize_result(name: str, result: BatchResult) -> dict:
    target = _target_indices(result.labels)
    pred = result.logits.argmax(dim=-1)
    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(result.logits, result.labels.float()).item())
    accuracy = float((pred == target).float().mean().item())
    return {
        "name": name,
        "sample_count": int(target.shape[0]),
        "loss_bce": round(loss, 6),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(_macro_f1(pred, target), 6),
        "per_class_accuracy": {key: round(value, 6) for key, value in _per_class_accuracy(pred, target).items()},
        "mean_logits": {
            key: round(float(result.logits[:, idx].mean().item()), 6)
            for idx, key in enumerate(CLASS_NAMES)
        },
    }


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    payload = load_feature_payload(args.features)
    tensors = load_training_tensors_v2(payload)
    indices = _build_eval_indices(tensors.labels.shape[0], args.eval_size, args.seed)

    image_embeddings = tensors.image_embeddings.index_select(0, indices)
    candidate_embeddings = tensors.candidate_embeddings.index_select(0, indices)
    prefix_embeddings = tensors.prefix_embeddings.index_select(0, indices)
    region_embeddings = tensors.region_embeddings.index_select(0, indices)
    labels = tensors.labels.index_select(0, indices).float()

    checkpoint_scorer = load_universal_scorer(args.psi_checkpoint, device=str(device))
    hardcoded_scorer = HardcodedUniversalScorer()
    zero_scorer = ZeroUniversalScorer()

    reports = []
    for name, scorer in (
        ("psi_checkpoint", checkpoint_scorer),
        ("hardcoded", hardcoded_scorer),
        ("zero", zero_scorer),
    ):
        result = evaluate_scorer(
            scorer,
            image_embeddings=image_embeddings,
            candidate_embeddings=candidate_embeddings,
            prefix_embeddings=prefix_embeddings,
            region_embeddings=region_embeddings,
            labels=labels,
            batch_size=args.batch_size,
            device=device,
        )
        reports.append(summarize_result(name, result))

    output = {
        "features": str(Path(args.features).resolve()),
        "psi_checkpoint": str(Path(args.psi_checkpoint).resolve()),
        "eval_size": int(labels.shape[0]),
        "seed": args.seed,
        "device": str(device),
        "reports": reports,
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
