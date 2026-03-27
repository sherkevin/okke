from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class IdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class ProjectorSpec:
    kind: str = "identity"
    checkpoint_path: str | None = None


def create_projector(
    spec: ProjectorSpec | None,
    input_dim: int,
    output_dim: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    spec = spec or ProjectorSpec()
    kind = spec.kind.lower()
    output_dim = output_dim or input_dim

    if kind == "identity":
        module: nn.Module = IdentityProjector()
    elif kind in {"linear", "loaded_linear"}:
        module = LinearProjector(input_dim, output_dim)
        if spec.checkpoint_path:
            ckpt = torch.load(spec.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported projector kind: {spec.kind}")

    if device is not None:
        module = module.to(device=device)
    if dtype is not None:
        module = module.to(dtype=dtype)
    module.eval()
    return module


def save_projector_checkpoint(module: nn.Module, output_path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": module.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, out)
    return out
