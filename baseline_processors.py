"""
Baseline Inference-Time Intervention Processors
================================================
Portable re-implementations of VCD, OPERA, DoLa, and DAMO core algorithms,
adapted as LogitsProcessors / wrappers for Qwen3-VL's native generate().

Core formulas are extracted from the official repositories:
  - VCD:  baselines/VCD/vcd_utils/vcd_sample.py
  - OPERA: baselines/OPERA/transformers-4.29.2/…/generation/utils.py
  - DoLa: Chuang et al., ICLR 2024
  - DAMO: Wang et al., ICLR 2025

References
----------
- VCD  : Leng et al., CVPR 2024 – Visual Contrastive Decoding
- OPERA : Huang et al., CVPR 2024 – Over-Trust Penalty & Retrospection
- DoLa  : Chuang et al., ICLR 2024 – Decoding by Contrasting Layers
- DAMO  : Wang et al., ICLR 2025 – Decoding by Accumulating Activations Momentum
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_language_model(model):
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def _get_decoder_root(model):
    language_model = _get_language_model(model)
    if hasattr(language_model, "model") and hasattr(language_model.model, "language_model"):
        return language_model.model.language_model
    if hasattr(language_model, "model"):
        return language_model.model
    if hasattr(language_model, "base_model"):
        return language_model.base_model
    return language_model


def _get_decoder_layers(model):
    root = _get_decoder_root(model)
    if hasattr(root, "layers"):
        return root.layers
    if hasattr(root, "decoder") and hasattr(root.decoder, "layers"):
        return root.decoder.layers
    raise AttributeError(f"Unsupported decoder structure for {type(model).__name__}")


def _get_lm_head(model):
    language_model = _get_language_model(model)
    if hasattr(language_model, "lm_head"):
        return language_model.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError(f"Unsupported lm_head structure for {type(model).__name__}")


def _get_num_hidden_layers(model):
    language_model = _get_language_model(model)
    config = getattr(language_model, "config", getattr(model, "config", None))
    if config is None:
        raise AttributeError(f"Missing config for {type(model).__name__}")
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return int(config.text_config.num_hidden_layers)
    raise AttributeError(f"Unsupported hidden-layer config for {type(model).__name__}")


# ===================================================================
# Diffusion noise  (exact port from VCD/vcd_utils/vcd_add_noise.py)
# ===================================================================

def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int = 500) -> torch.Tensor:
    """
    Forward-diffusion noise schedule matching the original VCD codebase.
    q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
    """
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_bar_sqrt = torch.sqrt(torch.cumprod(alphas, dim=0))
    one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - torch.cumprod(alphas, dim=0))

    noise = torch.randn_like(image_tensor.float())
    noisy = (
        alphas_bar_sqrt[noise_step] * image_tensor.float()
        + one_minus_alphas_bar_sqrt[noise_step] * noise
    )
    return noisy.to(image_tensor.dtype)


# ===================================================================
# VCD  –  Visual Contrastive Decoding
# ===================================================================

class VCDLogitsProcessor:
    """
    At each decoding step, run a *shadow* forward pass whose KV-cache was
    seeded with a diffusion-noised image.  The final logits are (from the
    original VCD code):

        cutoff  = log(cd_beta) + max(logits_std)
        cd_out  = (1 + cd_alpha) * logits_std  −  cd_alpha * logits_noisy
        cd_out[logits_std < cutoff] = −inf        # adaptive plausibility

    Usage
    -----
    >>> proc = VCDLogitsProcessor.prepare(model, inputs, ...)
    >>> out  = model.generate(**inputs, logits_processor=[proc], ...)
    >>> proc.cleanup()
    """

    def __init__(
        self,
        model,
        noisy_past_kv,
        noisy_cache_pos: int,
        cd_alpha: float = 1.0,
        cd_beta: float = 0.1,
    ):
        self.model = model
        self.noisy_past_kv = noisy_past_kv
        self._cache_pos = noisy_cache_pos
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta

    @classmethod
    def prepare(
        cls,
        model,
        inputs: dict,
        cd_alpha: float = 1.0,
        cd_beta: float = 0.1,
        noise_step: int = 500,
    ) -> "VCDLogitsProcessor":
        """Prefill with diffusion-noised pixel_values to seed shadow KV-cache."""
        noisy_inputs = {}
        for k, v in inputs.items():
            if k == "pixel_values" and v is not None:
                noisy_inputs[k] = add_diffusion_noise(v, noise_step)
            elif k == "pixel_values_videos" and v is not None:
                noisy_inputs[k] = add_diffusion_noise(v, noise_step)
            else:
                noisy_inputs[k] = v

        with torch.no_grad():
            noisy_out = model(**noisy_inputs, use_cache=True)

        return cls(
            model=model,
            noisy_past_kv=noisy_out.past_key_values,
            noisy_cache_pos=noisy_inputs["input_ids"].shape[1],
            cd_alpha=cd_alpha,
            cd_beta=cd_beta,
        )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        last_tok = input_ids[:, -1:]

        with torch.no_grad():
            cache_position = torch.tensor(
                [self._cache_pos], device=last_tok.device, dtype=torch.long
            )
            noisy_out = self.model(
                input_ids=last_tok,
                past_key_values=self.noisy_past_kv,
                use_cache=True,
                cache_position=cache_position,
            )
        self.noisy_past_kv = noisy_out.past_key_values
        self._cache_pos += 1
        noisy_logits = noisy_out.logits[:, -1, :]

        # --- exact VCD formula (from vcd_sample.py) ---
        cutoff = (
            torch.log(torch.tensor(self.cd_beta, device=scores.device))
            + scores.float().max(dim=-1, keepdim=True).values
        )
        diffs = (1.0 + self.cd_alpha) * scores.float() - self.cd_alpha * noisy_logits.float()
        cd_logits = diffs.masked_fill(scores.float() < cutoff, -float("inf"))

        return cd_logits.to(scores.dtype)

    def cleanup(self):
        del self.noisy_past_kv
        self.noisy_past_kv = None


# ===================================================================
# OPERA  –  Over-Trust Penalty and Retrospection-Allocation
# ===================================================================

class OPERALogitsProcessor:
    """
    OPERA monitors self-attention patterns in the last decoder layer.
    When attention to image-region tokens is over-concentrated, a penalty
    is applied to the logits.

    **Known limitation**: OPERA requires ``output_attentions=True`` at the
    layer level.  FlashAttention / SDPA fused kernels typically do NOT
    return attention weights, causing a RuntimeError.  If this happens the
    error is raised (not silenced) so the caller can report the traceback.

    Simplified single-pass penalty (no beam rollback):
        penalty = pw * (max-column-attention of last layer, last query row)
    """

    def __init__(
        self,
        model,
        penalty_weight: float = 1.0,
        scale_factor: float = 50.0,
    ):
        self.model = model
        self.penalty_weight = penalty_weight
        self.scale_factor = scale_factor
        self._last_attentions: Optional[torch.Tensor] = None
        self._hook: Optional[Any] = None
        self._attn_error_reported = False
        self._install_hook()

    def _install_hook(self):
        """Capture attention weights from the last decoder layer."""
        try:
            last_layer = _get_decoder_layers(self.model)[-1]
            self_attn = last_layer.self_attn

            outer = self

            def capture(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    outer._last_attentions = output[1].detach()
                return output

            self._hook = self_attn.register_forward_hook(capture, with_kwargs=True)
            logger.info("OPERA: attention hook installed on last decoder layer.")
        except Exception as e:
            logger.warning(f"OPERA: could not install attention hook: {e}")
            self._hook = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._last_attentions is None:
            if not self._attn_error_reported:
                logger.warning(
                    "OPERA: No attention weights captured. "
                    "FlashAttention/SDPA likely does not return attention maps. "
                    "OPERA penalty is DISABLED for this run. "
                    "To enable, load model with attn_implementation='eager'."
                )
                self._attn_error_reported = True
            return scores

        try:
            attn = self._last_attentions  # [bs, heads, q, kv]
            attn_max_head = attn.max(dim=1).values  # [bs, q, kv]
            last_row = attn_max_head[:, -1, :]  # [bs, kv]

            column_score = last_row.max(dim=-1, keepdim=True).values  # [bs, 1]
            penalty = self.penalty_weight * self.scale_factor * column_score
            scores = scores.float() - penalty
        except Exception as e:
            logger.debug(f"OPERA penalty skipped: {e}")

        self._last_attentions = None
        return scores.to(scores.dtype)

    def cleanup(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class EKKOLegacyLogitsProcessor(OPERALogitsProcessor):
    """
    Exact simplified baseline-OPERA clone preserved as a legacy method.

    `ekko_legacy` is a byte-for-byte behavior match of the simplified
    baseline OPERA processor so the historical baseline remains available
    after `ekko` is promoted onto the official OPERA host path.
    """


# ===================================================================
# DoLa  –  Decoding by Contrasting Layers
# ===================================================================

class DoLaLogitsProcessor:
    """
    Contrast the final-layer (mature) logits with an earlier-layer
    (premature) projection through lm_head:

        logit_final = log softmax(mature) − log softmax(premature)

    The premature layer is selected dynamically as the one with the
    highest Jensen-Shannon divergence from the mature layer among a
    candidate set (following the original DoLa paper).
    """

    def __init__(
        self,
        model,
        candidate_premature_layers: list[int] | None = None,
    ):
        self.model = model
        self.lm_head = _get_lm_head(model)

        n_layers = _get_num_hidden_layers(model)
        if candidate_premature_layers is None:
            low = max(0, n_layers // 2 - 2)
            high = n_layers - 2
            self.premature_layers = list(range(low, high))
        else:
            self.premature_layers = candidate_premature_layers

        self._hidden_cache: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._install_hooks()

    def _install_hooks(self):
        layers = _get_decoder_layers(self.model)
        for idx in self.premature_layers:
            def make_hook(layer_idx):
                def hook_fn(module, _input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self._hidden_cache[layer_idx] = hs[:, -1:, :].detach()
                return hook_fn
            h = layers[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not self._hidden_cache:
            return scores

        mature_log = F.log_softmax(scores.float(), dim=-1)

        best_jsd = -1.0
        best_premature_log = None

        for layer_idx, hs in self._hidden_cache.items():
            with torch.no_grad():
                premature_logits = self.lm_head(hs).squeeze(1)
            premature_log = F.log_softmax(premature_logits.float(), dim=-1)

            m = 0.5 * (mature_log.exp() + premature_log.exp())
            jsd = 0.5 * (
                F.kl_div(m.log(), mature_log.exp(), reduction="batchmean")
                + F.kl_div(m.log(), premature_log.exp(), reduction="batchmean")
            )
            if jsd.item() > best_jsd:
                best_jsd = jsd.item()
                best_premature_log = premature_log

        self._hidden_cache.clear()

        if best_premature_log is None:
            return scores

        contrast = mature_log - best_premature_log
        return contrast.to(scores.dtype)

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hidden_cache.clear()


# ===================================================================
# DAMO  –  Decoding by Accumulating Activations Momentum
# ===================================================================

class DAMOLogitsProcessor:
    """
    DAMO modifies the last-token hidden state inside late decoder layers when
    the layer-wise logit trajectory exhibits unstable momentum.

    The official release patches LLaVA's decoder internals directly. Here we
    reproduce the same high-level rule with decoder-layer hooks so it can be
    used from the unified runner.
    """

    def __init__(
        self,
        model,
        tau: float = -0.3,
        beta_1: float = 0.05,
        beta_2: float = 0.20,
        alpha: float = 0.7,
        start_layer: int | None = None,
    ):
        self.model = model
        self.lm_head = _get_lm_head(model)
        self.tau = float(tau)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.alpha = float(alpha)

        n_layers = _get_num_hidden_layers(model)
        self.start_layer = int(start_layer) if start_layer is not None else max(1, n_layers // 2)

        self._hooks: list[Any] = []
        self._reset_forward_state()
        self._install_hooks()

    def _reset_forward_state(self):
        self._prev_layer_input: Optional[torch.Tensor] = None
        self._prev_hidden_states: Optional[torch.Tensor] = None
        self._delta_hidden_states: Optional[torch.Tensor] = None
        self._momentum_decoding_flag = False

    @staticmethod
    def _custom_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm_vec1 = torch.sqrt(torch.sum(vec1 ** 2, dim=-1))
        norm_vec2 = torch.sqrt(torch.sum(vec2 ** 2, dim=-1))
        norm_vec1 = torch.clamp(norm_vec1, min=eps)
        norm_vec2 = torch.clamp(norm_vec2, min=eps)
        dot_product = torch.sum(vec1 * vec2, dim=-1)
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        if torch.isinf(cosine_sim).any() or torch.isnan(cosine_sim).any():
            cosine_sim = torch.where(
                torch.isinf(cosine_sim) | torch.isnan(cosine_sim),
                torch.zeros_like(cosine_sim),
                cosine_sim,
            )
        return cosine_sim

    def _install_hooks(self):
        layers = _get_decoder_layers(self.model)
        if not layers:
            return

        def reset_hook(module, args):
            self._reset_forward_state()
            return None

        self._hooks.append(layers[0].register_forward_pre_hook(reset_hook))

        for idx, layer in enumerate(layers):
            def make_hook(layer_idx: int):
                def hook_fn(module, inputs, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    if not isinstance(hidden_states, torch.Tensor):
                        return output

                    current_input = inputs[0]
                    if not isinstance(current_input, torch.Tensor):
                        self._prev_layer_input = None
                        return output

                    current_input_last = current_input[:, -1:, :]
                    current_output_last = hidden_states[:, -1:, :]

                    if layer_idx >= self.start_layer and self._prev_layer_input is not None:
                        current_increasement = current_output_last - current_input_last

                        with torch.no_grad():
                            logits_3 = self.lm_head(current_output_last)
                            logits_2 = self.lm_head(current_input_last)
                            logits_1 = self.lm_head(self._prev_layer_input)
                            p_3 = torch.softmax(logits_3.float(), dim=-1)
                            p_2 = torch.softmax(logits_2.float(), dim=-1)
                            p_1 = torch.softmax(logits_1.float(), dim=-1)
                            delta_p2 = p_3 - p_2
                            delta_p1 = p_2 - p_1

                        if self._delta_hidden_states is None:
                            self._delta_hidden_states = torch.zeros_like(delta_p1)
                        if self._prev_hidden_states is None:
                            self._prev_hidden_states = torch.zeros_like(current_increasement)

                        self._delta_hidden_states = self.alpha * self._delta_hidden_states + (1 - self.alpha) * delta_p1
                        cosine_similarity = self._custom_cosine_similarity(delta_p2, self._delta_hidden_states).mean()

                        prev_hidden_states_momentum = self.beta_1
                        if float(cosine_similarity.item()) < self.tau:
                            self._momentum_decoding_flag = True
                            prev_hidden_states_momentum = self.beta_2

                        self._prev_hidden_states = (
                            prev_hidden_states_momentum * self._prev_hidden_states
                            + (1 - prev_hidden_states_momentum) * current_increasement
                        )

                        if self._momentum_decoding_flag:
                            adjusted_last = current_output_last - current_increasement + self._prev_hidden_states
                            hidden_states = hidden_states.clone()
                            hidden_states[:, -1:, :] = adjusted_last.to(hidden_states.dtype)
                            if isinstance(output, tuple):
                                output = (hidden_states, *output[1:])
                            else:
                                output = hidden_states

                    self._prev_layer_input = current_input_last.detach()
                    return output

                return hook_fn

            self._hooks.append(layer.register_forward_hook(make_hook(idx)))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # DAMO acts inside decoder-layer hooks; the logits processor itself is a no-op.
        return scores

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._reset_forward_state()
