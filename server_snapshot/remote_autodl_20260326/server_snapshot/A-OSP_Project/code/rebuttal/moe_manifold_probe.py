"""
Appendix H — MoE Topological Fragmentation (manifold probe) — STUB / PREP ONLY
==============================================================================
Paper claim: MoE multimodal backbones exhibit *topological fragmentation* between
shared vs routed expert pathways. This script is a **hook scaffold** for Agent 1 to
run when GPU is fully free.

Model (registered): **deepseek-ai/deepseek-vl2-tiny** (~1.0B activated params,
DeepSeekMoE backbone; n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6).

Note: `deepseek-ai/deepseek-vl-1.3b-moe` does **not** exist on Hugging Face; VL2-Tiny
is the supported lightweight MoE vision-language checkpoint.

Upstream runtime (required before running — not executed in this stub):
  git clone https://github.com/deepseek-ai/DeepSeek-VL2.git && cd DeepSeek-VL2 && pip install -e .

Inference pattern (from upstream README): load `DeepseekVLV2ForCausalLM` with
`trust_remote_code=True`, then use `language_model` for MoE blocks.

Planned outputs (Agent 1): JSON under `logs/rebuttal/moe_manifold_probe.json` with
per-layer statistics on shared vs routed expert activations / router weights —
supporting Appendix H figures/tables (exact figure IDs to be bound when runs complete).

Do **not** invoke `main()` from automation until DeepSeek-VL2 package is installed
and VRAM is available.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
DEFAULT_MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"
OUT_JSON = PROJECT / "logs" / "rebuttal" / "moe_manifold_probe.json"


# --- discovery heuristics (tune after inspecting named_modules() on real load) ---

SHARED_NAME_HINTS = ("shared", "shared_expert", "shared_mlp", "sharedExperts")
ROUTED_NAME_HINTS = ("experts", "routed", "moe", "expert_mlps", "gate")
ROUTER_NAME_HINTS = ("gate", "router", "route")


def _name_hits(name: str, hints: Tuple[str, ...]) -> bool:
    ln = name.lower()
    return any(h in ln for h in hints)


@dataclass
class ExpertHookState:
    """Buffers for forward-hook captures (per module)."""

    shared_outputs: List[torch.Tensor] = field(default_factory=list)
    routed_outputs: List[torch.Tensor] = field(default_factory=list)
    router_logits: List[torch.Tensor] = field(default_factory=list)


def classify_moe_submodule(name: str, module: nn.Module) -> Optional[str]:
    """
    Bucket a submodule for hook registration.
    Returns 'shared' | 'routed' | 'router' | None
    """
    if not isinstance(module, nn.Module):
        return None
    if _name_hits(name, ROUTER_NAME_HINTS) and "expert" not in name.lower():
        # gates often named *gate*; avoid grabbing unrelated 'aggregate' etc.
        if list(module.parameters()):
            return "router"
    if _name_hits(name, SHARED_NAME_HINTS):
        return "shared"
    # routed expert FFNs: typical DeepSeekMoE uses `experts` ModuleList
    if _name_hits(name, ROUTED_NAME_HINTS) and "shared" not in name.lower():
        if "expert" in name.lower() or "mlp" in name.lower():
            return "routed"
    return None


def register_moe_hooks(
    root: nn.Module,
    state: Optional[ExpertHookState] = None,
) -> Tuple[ExpertHookState, List[torch.utils.hooks.RemovableHandle]]:
    """
    Register forward hooks on language-model MoE submodules (shared vs routed vs router).

    Hooks store detached CPU copies to limit VRAM; Agent 1 should trim / aggregate.
    """
    if state is None:
        state = ExpertHookState()
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(
        bucket: str,
    ) -> Callable[[nn.Module, Any, Any], None]:
        def hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            if bucket == "shared":
                state.shared_outputs.append(t.detach().cpu().float())
            elif bucket == "routed":
                state.routed_outputs.append(t.detach().cpu().float())
            elif bucket == "router":
                state.router_logits.append(t.detach().cpu().float())

        return hook

    for name, mod in root.named_modules():
        if name == "":
            continue
        bucket = classify_moe_submodule(name, mod)
        if bucket is None:
            continue
        handles.append(mod.register_forward_hook(_make_hook(bucket)))

    return state, handles


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        h.remove()


def resolve_snapshot_dir(model_id: str) -> Path:
    """Resolve HF hub snapshot directory for documentation / local path checks."""
    try:
        from huggingface_hub import snapshot_download

        p = snapshot_download(model_id, local_files_only=True)
        return Path(p)
    except Exception:
        return Path("(not cached — run snapshot_download without local_files_only)")


def build_model_stub(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Any]:
    """
    Load DeepSeek-VL2 multimodal MoE model (requires `deepseek_vl` from DeepSeek-VL2 repo).

    Returns (full_vl_model, language_model_submodule) for hooking experts inside LM.
    """
    # Deferred import — package not present until upstream `pip install -e .`
    from transformers import AutoModelForCausalLM  # type: ignore

    vl = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        vl = vl.to(device)
    lm = getattr(vl, "language_model", None)
    if lm is None:
        raise RuntimeError("Expected attribute `language_model` on DeepSeek-VL2 wrapper.")
    vl.eval()
    return vl, lm


def run_probe_forward_stub(
    language_model: nn.Module,
    dummy_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Placeholder forward: Agent 1 must replace with real inputs_embeds + masks from
    DeepseekVLV2Processor (image + text). For now, documents the intended call site.
    """
    raise NotImplementedError(
        "Agent 1: use vl_gpt.prepare_inputs_embeds(**prepare_inputs) then "
        "language_model forward / generate per DeepSeek-VL2 README."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE manifold probe (Appendix H)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load model + register hooks (requires DeepSeek-VL2 `pip install -e .` + GPU).",
    )
    args = parser.parse_args()

    snap = resolve_snapshot_dir(args.model_id)
    print(json.dumps({"model_id": args.model_id, "local_snapshot": str(snap)}, indent=2))

    if not args.execute:
        print(
            "[moe_manifold_probe] prep-only: install DeepSeek-VL2 (`pip install -e .` from "
            "https://github.com/deepseek-ai/DeepSeek-VL2 ), then run with --execute when GPU is free."
        )
        return

    # Real execution path (Agent 1)
    vl, lm = build_model_stub(args.model_id)
    state, handles = register_moe_hooks(lm)
    try:
        # run_probe_forward_stub(lm, ...)  # replace with real forward
        raise RuntimeError("Forward not implemented in stub — implement by Agent 1.")
    finally:
        remove_hooks(handles)


if __name__ == "__main__":
    main()
