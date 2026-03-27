"""
A-OSP Hook Interface — Entry Point for Agent 2 (Evaluation Engineer)
====================================================================
Provides a single-call API:

    apply_aosp_hook(model, v_matrix_path) -> AOSPHandle

This loads the pre-extracted bias subspace V_matrix, wires up the adaptive
intervention hook on the correct layer, and pre-warms the compiled kernels.
Agent 2 calls this once before running any benchmark.

Usage:
    from aosp_hook import apply_aosp_hook

    handle = apply_aosp_hook(model, "/path/to/V_matrix.pt")
    # ... run benchmarks with model.generate() ...
    handle.remove()   # clean up when done
"""

import torch
from adaptive_intervention import (
    AOSPConfig,
    AOSPState,
    build_aosp_hook,
    ortho_project_rescale,
    compute_projection_energy,
)


class AOSPHandle:
    """Wrapper that bundles the hook handle with intervention state for diagnostics."""

    def __init__(self, hook_handle, state: AOSPState, layer_idx: int):
        self._hook_handle = hook_handle
        self.state = state
        self.layer_idx = layer_idx

    def reset(self):
        """Reset per-sequence state. Call before each model.generate()."""
        self.state.reset()

    def remove(self):
        """Unregister the hook from the model."""
        self._hook_handle.remove()

    @property
    def intervention_count(self) -> int:
        return self.state.interventions


def apply_aosp_hook(
    model,
    v_matrix_path: str,
    *,
    alpha: float = 0.5,
    mu: float = 1.5,
    beta: float = 0.9,
    epsilon_steady: float = 0.1,
    layer_idx: int | None = None,
    dynamic_mu: bool = False,
) -> AOSPHandle:
    """
    One-call setup for A-OSP inference-time intervention.

    Args:
        model: A loaded Qwen2VLForConditionalGeneration (or compatible).
        v_matrix_path: Path to the V_matrix.pt saved by subspace_extractor.
        alpha, mu, beta, epsilon_steady: A-OSP hyperparameters (defaults match
            the paper's global fixed config).
        layer_idx: Override the intervention layer (0-indexed).  When None,
            uses the layer_idx stored in the checkpoint.  Enables the
            layer-sensitivity scan (Appendix C) to sweep across layers
            with a single V_matrix.

    Returns:
        AOSPHandle with .reset() / .remove() / .intervention_count
    """
    checkpoint = torch.load(v_matrix_path, map_location="cpu", weights_only=True)
    V_bias = checkpoint["V_bias"]                       # [K, D]
    layer_idx = layer_idx if layer_idx is not None else checkpoint["layer_idx"]
    K = checkpoint["K"]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    V_bias = V_bias.to(device=device, dtype=dtype)

    cfg = AOSPConfig(
        alpha=alpha,
        mu=mu,
        beta=beta,
        epsilon_steady=epsilon_steady,
        K=K,
        L_prior=checkpoint.get("L_prior", 1.0),
        dynamic_mu=dynamic_mu,
    )
    state = AOSPState(cfg)

    # Resolve lm_head for entropy-based burn-in
    lm_head = getattr(model, 'lm_head', None)

    hook_fn = build_aosp_hook(V_bias, cfg, state, lm_head=lm_head)
    # Qwen2-VL nests decoder layers under model.language_model.layers
    if hasattr(model.model, 'language_model'):
        decoder_layers = model.model.language_model.layers
    else:
        decoder_layers = model.model.layers
    handle = decoder_layers[layer_idx].register_forward_hook(hook_fn)

    # ---- Pre-warm compiled kernels (critical for RTX 5090 latency) ----
    _prewarm_compiled_kernels(V_bias, device, dtype)

    print(f"[A-OSP] Hook installed on layer {layer_idx} | K={K} | "
          f"alpha={alpha} mu={mu} beta={beta} eps={epsilon_steady}")
    print(f"[A-OSP] EVR of bias subspace: {checkpoint.get('evr', 'N/A')}")

    return AOSPHandle(handle, state, layer_idx)


def _prewarm_compiled_kernels(V_bias: torch.Tensor, device, dtype):
    """
    Pre-warm torch.compile graphs with dummy tensors to avoid
    graph recompilation during actual inference.
    """
    D = V_bias.shape[1]
    dummy_H = torch.randn(1, D, device=device, dtype=dtype)
    V_dev = V_bias.to(device=device, dtype=dtype)

    for _ in range(3):
        _ = compute_projection_energy(dummy_H, V_dev)
        _ = ortho_project_rescale(dummy_H, V_dev, 0.5)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("[A-OSP] Compiled kernels pre-warmed.")
