#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bra_projector import LinearProjector, save_projector_checkpoint


def info_nce_loss(queries: torch.Tensor, positives: torch.Tensor, temperature: float) -> torch.Tensor:
    q = F.normalize(queries, dim=-1)
    p = F.normalize(positives, dim=-1)
    logits = q @ p.T / max(temperature, 1e-6)
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def load_feature_tensors(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "vision_features" not in payload or "token_embeddings" not in payload:
        raise ValueError(
            "Training stub expects a .pt file with keys `vision_features` and `token_embeddings`."
        )
    vision = payload["vision_features"].float()
    tokens = payload["token_embeddings"].float()
    if vision.ndim != 2 or tokens.ndim != 2 or vision.shape[0] != tokens.shape[0]:
        raise ValueError("Feature tensors must be 2D and aligned on the batch dimension.")
    return vision, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to .pt feature file")
    parser.add_argument("--output", required=True, help="Projector checkpoint output path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    vision, tokens = load_feature_tensors(args.features)
    dataset = TensorDataset(vision, tokens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projector = LinearProjector(vision.shape[-1], tokens.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        count = 0
        for v_batch, t_batch in loader:
            v_batch = v_batch.to(device)
            t_batch = t_batch.to(device)
            proj = projector(v_batch)
            loss = info_nce_loss(proj, t_batch, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * v_batch.shape[0]
            count += v_batch.shape[0]
        print(f"epoch={epoch} loss={running / max(count, 1):.6f}")

    out = save_projector_checkpoint(
        projector,
        args.output,
        extra={
            "input_dim": vision.shape[-1],
            "output_dim": tokens.shape[-1],
            "features_path": str(args.features),
        },
    )
    print(f"Saved projector checkpoint to {out}")


if __name__ == "__main__":
    main()
