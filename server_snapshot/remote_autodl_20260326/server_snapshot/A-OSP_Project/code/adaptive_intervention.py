"""
A-OSP Adaptive Orthogonal Subspace Projection — Inference-Time Intervention
===========================================================================
Implements the real-time dynamic intervention logic:
  1. Shannon-entropy-based adaptive burn-in  (replaces hard-coded N=5)
  2. L2 projection energy monitoring with conditional EMA baseline
  3. Scale-preserved soft orthogonal projection
  4. Static compiled kernel to fuse projection / norm / rescale across the
     RTX 5090 memory-bandwidth wall.

RED LINES:
  - The compiled projection function is a PURE STATIC function extracted
    outside the hook. It must be pre-warmed at model load time.
    NEVER call torch.compile inside the hook — it triggers graph
    recompilation and latency explosion.
  - All norm denominators are clamped to 1e-6 to prevent NaN.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

# ========================== Hyperparameters ==========================
@dataclass
class AOSPConfig:
    alpha: float = 0.5           # projection strength
    mu: float = 1.5              # EMA exceedance trigger multiplier
    beta: float = 0.9            # EMA decay rate
    epsilon_steady: float = 0.1  # entropy-diff threshold for burn-in end
    K: int = 20                  # subspace rank (must match V_matrix)
    L_prior: float = 1.0        # frozen baseline during burn-in (calibrated offline)
    dynamic_mu: bool = False     # Use visual entropy/variance to dynamically adjust mu


# ============ Static Compiled Kernel (fuse for RTX 5090) ============

def _ortho_project_rescale(
    H: torch.Tensor,          # [batch, D]  current hidden state
    V: torch.Tensor,           # [K, D]     bias subspace basis
    alpha: float,
) -> torch.Tensor:
    """
    Scale-preserved soft orthogonal projection.
      H_proj = H - alpha * sum_i <H, v_i> v_i
      H'     = H_proj / ||H_proj|| * ||H||
    Fused into one graph to minimize HBM round-trips on RTX 5090.
    """
    orig_norm = torch.clamp(H.norm(dim=-1, keepdim=True), min=1e-6)      # [B, 1]
    projections = H @ V.T                                                 # [B, K]
    H_proj = H - alpha * (projections @ V)                                # [B, D]
    proj_norm = torch.clamp(H_proj.norm(dim=-1, keepdim=True), min=1e-6)  # [B, 1]
    return H_proj / proj_norm * orig_norm                                 # [B, D]


import os as _os

_compile_mode = _os.environ.get("AOSP_COMPILE_MODE", "reduce-overhead")
if _compile_mode == "off":
    ortho_project_rescale = _ortho_project_rescale
else:
    ortho_project_rescale = torch.compile(
        _ortho_project_rescale, mode=_compile_mode
    )


# ============ Projection Energy Calculator (also compiled) ============

def _compute_projection_energy(
    H: torch.Tensor,    # [batch, D]
    V: torch.Tensor,    # [K, D]
) -> torch.Tensor:
    """
    L_t = sqrt( sum_i <H_norm, v_i>^2 )
    L2 norm of the projection onto S_bias — physical "energy" measure.
    """
    proj = H @ V.T                                             # [B, K]
    return torch.sqrt((proj ** 2).sum(dim=-1, keepdim=True))   # [B, 1]


if _compile_mode == "off":
    compute_projection_energy = _compute_projection_energy
else:
    compute_projection_energy = torch.compile(
        _compute_projection_energy, mode=_compile_mode
    )


# ========================== Intervention State ==========================

class AOSPState:
    """Mutable per-sequence state for adaptive intervention."""

    def __init__(self, cfg: AOSPConfig):
        self.cfg = cfg
        self.t: int = 0                         # current generation step
        self.N_adaptive: int | None = None       # burn-in end step (None = still burning in)
        self.L_bar: float = cfg.L_prior          # EMA baseline
        self.prev_entropy: float | None = None   # for delta-entropy tracking
        self.interventions: int = 0              # counter for diagnostics
        self.visual_variance: float = 10.0       # visual token variance
        self.mu_eff: float = cfg.mu              # effective trigger multiplier

    def reset(self):
        self.t = 0
        self.N_adaptive = None
        self.L_bar = self.cfg.L_prior
        self.prev_entropy = None
        self.interventions = 0
        self.visual_variance = 10.0
        self.mu_eff = self.cfg.mu


# ========================== Core Hook Logic ==========================

def build_aosp_hook(V_bias: torch.Tensor, cfg: AOSPConfig, state: AOSPState, *, lm_head=None):
    """
    Returns a forward_hook closure for the target transformer layer.

    At each generation step:
      1. Check entropy-based burn-in status.
      2. Compute L2 projection energy L_t.
      3. If burn-in complete AND L_t > mu * L_bar → trigger intervention.
      4. Update conditional EMA baseline (only on non-spike steps).

    Args:
        lm_head: The model's lm_head module for computing logits during
                 entropy-based burn-in. If None, falls back to module traversal.
    """
    V = V_bias.clone()           # [K, D]  stays on model device

    def hook_fn(module, inp, out):
        # Qwen2-VL / Qwen2.5-VL returns a tuple: (hidden_states, past_kv, ...)
        # Qwen3-VL with FA2 returns a bare Tensor: [batch, seq_len, D]
        if isinstance(out, tuple):
            hidden = out[0]          # [batch, seq_or_1, D]
        else:
            hidden = out             # [batch, seq_or_1, D]  (Qwen3-VL FA2)

        if hidden.shape[1] != 1:
            if cfg.dynamic_mu:
                # Calculate variance of prefill hidden states as a proxy for visual clarity
                # Ambiguous images tend to have lower variance
                state.visual_variance = hidden[0].var(dim=0).mean().item()
                # If variance is low, increase mu to prevent over-penalization
                baseline_var = 1e5  # aggressively high to ensure it triggers
                state.mu_eff = cfg.mu * max(1.0, baseline_var / max(state.visual_variance, 1e-6))
                # print(f"Dynamic mu: {state.mu_eff:.2f}, variance: {state.visual_variance:.2f}")
            else:
                state.mu_eff = cfg.mu
            return               # skip prefill pass

        state.t += 1
        H = hidden[:, -1, :]    # [batch, D]

        # --- Step 1: Entropy-aware dynamic burn-in ---
        if state.N_adaptive is None:
            logits = _peek_logits(module, H, lm_head=lm_head)
            if logits is not None:
                ent = _shannon_entropy(logits)
                if state.prev_entropy is not None:
                    delta_e = abs(ent - state.prev_entropy)
                    if delta_e < cfg.epsilon_steady:
                        state.N_adaptive = state.t
                state.prev_entropy = ent
                # Short-generation bypass: if model is highly confident (low entropy),
                # treat burn-in as complete immediately.  This handles binary tasks
                # (POPE yes/no) where only 1 decode step occurs so entropy never
                # gets a chance to stabilize via delta comparison.
                if state.N_adaptive is None and ent < 0.05:
                    state.N_adaptive = state.t
            if state.N_adaptive is None:
                return           # still in burn-in → no intervention

        # --- Step 2: Projection energy ---
        V_dev = V.to(H.device, dtype=H.dtype)
        L_t = compute_projection_energy(H, V_dev).item()

        # --- Step 3: Trigger decision ---
        if L_t > state.mu_eff * state.L_bar:
            H_new = ortho_project_rescale(H, V_dev, cfg.alpha)
            hidden[:, -1, :] = H_new
            state.interventions += 1
        else:
            # --- Step 4: Conditional EMA update (non-spike only) ---
            if state.t > (state.N_adaptive or 0):
                state.L_bar = cfg.beta * state.L_bar + (1.0 - cfg.beta) * L_t

    return hook_fn


# ========================== Utility ==========================

def _shannon_entropy(logits: torch.Tensor) -> float:
    """H(P) = -sum P(x) log P(x)  over vocabulary."""
    probs = F.softmax(logits[:, -1, :], dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=-1).mean().item()


def _peek_logits(layer_module, H: torch.Tensor, *, lm_head=None):
    """
    Compute logits from the hidden state for entropy calculation.
    Uses the directly-passed lm_head (preferred) or falls back to
    module tree traversal.
    """
    try:
        head = lm_head
        if head is None:
            model = layer_module
            while hasattr(model, '_modules') and not hasattr(model, 'lm_head'):
                if hasattr(model, '_parent'):
                    model = model._parent
                else:
                    return None
            head = getattr(model, 'lm_head', None)

        if head is not None:
            with torch.no_grad():
                return head(H.unsqueeze(1))
    except Exception:
        pass
    return None
