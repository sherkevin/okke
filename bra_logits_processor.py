"""
Canonical internal TLRA/BRA LogitsProcessor implementation.

This file is the source of truth for the model-coupled intervention path that
reads model-native visual hidden states and reweights next-token logits through
`lm_head.weight` geometry. It remains valuable as an internal baseline, but it
is not the architectural home for a strict train-once, run-anywhere universal
plugin claim. That universal route now lives in `bra_universal_plugin.py`.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from bra_projector import ProjectorSpec, create_projector
from bra_vasm import ProbabilisticVASM, detect_tokenizer_family, is_continuation_subword, normalize_token

logger = logging.getLogger(__name__)


@dataclass
class BRAStats:
    decode_steps: int = 0
    eligible_steps: int = 0
    intervention_steps: int = 0
    candidate_window_sum: int = 0
    visual_topk_sum: int = 0
    resonance_time_ms_sum: float = 0.0
    routing_time_ms_sum: float = 0.0
    vasm_time_ms_sum: float = 0.0
    continuation_attempts: int = 0
    continuation_successes: int = 0
    suffix_collapse_failures: int = 0
    selected_frame_histogram: dict[int, int] = field(default_factory=dict)
    continuation_failure_examples: list[dict[str, Any]] = field(default_factory=list)
    visual_state_provenance: dict[str, Any] = field(default_factory=dict)
    vasm_metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.decode_steps = 0
        self.eligible_steps = 0
        self.intervention_steps = 0
        self.candidate_window_sum = 0
        self.visual_topk_sum = 0
        self.resonance_time_ms_sum = 0.0
        self.routing_time_ms_sum = 0.0
        self.vasm_time_ms_sum = 0.0
        self.continuation_attempts = 0
        self.continuation_successes = 0
        self.suffix_collapse_failures = 0
        self.selected_frame_histogram.clear()
        self.continuation_failure_examples.clear()
        self.visual_state_provenance.clear()
        self.vasm_metadata.clear()

    def to_dict(self) -> dict[str, Any]:
        eligible = max(self.eligible_steps, 1)
        return {
            **asdict(self),
            "avg_candidate_window": self.candidate_window_sum / eligible,
            "avg_visual_topk": self.visual_topk_sum / eligible,
            "avg_resonance_time_ms": self.resonance_time_ms_sum / eligible,
            "avg_routing_time_ms": self.routing_time_ms_sum / eligible,
            "avg_vasm_time_ms": self.vasm_time_ms_sum / eligible,
            "intervention_rate": self.intervention_steps / eligible,
            "continuation_success_rate": self.continuation_successes / max(self.continuation_attempts, 1),
            "selected_frame_histogram": dict(sorted(self.selected_frame_histogram.items())),
            "continuation_failure_examples": list(self.continuation_failure_examples),
            "visual_state_provenance": dict(self.visual_state_provenance),
            "vasm_metadata": dict(self.vasm_metadata),
        }


@dataclass
class BRAVisionMeta:
    n_image_tokens: int = 0
    frame_indices: Optional[torch.Tensor] = None
    n_visual_tokens: int = 0
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class BRAConfig:
    mode: str = "bra_zero"
    alpha: float = 0.30
    beta: float = 0.15
    epsilon: float = 0.05
    top_k: int = 50
    warmup_steps: int = 3
    rho: float = 0.01
    visual_topk_min: int = 2
    visual_topk_max: int = 64
    tau_sim: float = 0.07
    candidate_temperature: float = 0.25
    lambda_static: float = 1.0
    lambda_temporal: float = 0.0
    use_temporal_diff_branch: bool = False
    projector_kind: str = "identity"
    projector_checkpoint: Optional[str] = None
    mask_variant: str = "vasm"
    vasm_artifact_path: Optional[str] = None
    apply_entropy_trigger: bool = False
    apply_margin_trigger: bool = False
    apply_evidence_trigger: bool = False
    require_joint_trigger: bool = False
    margin_epsilon: float = 0.18
    resonance_floor: float = 0.10
    resonance_gap: float = 0.12
    legacy_relative_mix: float = 0.50
    legacy_use_abs_logit_scale: bool = True
    aggregation_mode: Optional[str] = None
    random_patch_seed: int = 1234
    record_diagnostics: bool = True
    audit_max_steps: int = 32
    debug: bool = False


class BRAVisionExtractor:
    def __init__(self, model, adapter):
        self.adapter = adapter
        self.model = model
        self._hidden_buf: Optional[torch.Tensor] = None
        inner = adapter.get_inner_model(model)
        self._hook = inner.register_forward_hook(self._capture)
        if hasattr(adapter, "describe_visual_state_source"):
            self.source = adapter.describe_visual_state_source(model)
        else:
            self.source = {
                "module_path": type(inner).__name__,
                "capture_kind": "forward_output_last_hidden_state",
                "layer_role": "final",
            }

    def _capture(self, module, _input, output):
        if isinstance(output, tuple):
            self._hidden_buf = output[0]
        else:
            self._hidden_buf = getattr(output, "last_hidden_state", output)

    @torch.no_grad()
    def extract_vision_features(
        self,
        input_ids: torch.LongTensor,
        image_token_id: int,
        video_token_id: Optional[int] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        build_temporal_diff: bool = False,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], BRAVisionMeta]:
        hs = self._hidden_buf
        meta = BRAVisionMeta()
        meta.source = dict(self.source)
        if hs is None:
            return None, None, meta

        vision_mask = input_ids == image_token_id
        if video_token_id is not None:
            vision_mask = vision_mask | (input_ids == video_token_id)
        vision_mask = vision_mask.to(hs.device)

        if not vision_mask.any():
            if self.adapter.name == "instructblip":
                nq = getattr(self.model.config, "num_query_tokens", 32)
                if hs.shape[1] > nq:
                    z = hs[0, :nq].detach().clone()
                    meta.n_visual_tokens = z.shape[0]
                    self._hidden_buf = None
                    return z, None, meta
            self._hidden_buf = None
            return None, None, meta

        z_all = hs[0, vision_mask[0]].detach().clone()
        self._hidden_buf = None
        meta.n_visual_tokens = z_all.shape[0]

        z_delta = None
        if video_token_id is not None and video_grid_thw is not None and self.adapter.supports_video():
            meta = self._build_visual_meta(z_all, input_ids, image_token_id, video_token_id, video_grid_thw)
            if build_temporal_diff:
                z_delta = self._build_temporal_diff(z_all, input_ids, video_token_id, image_token_id, video_grid_thw)
        return z_all, z_delta, meta

    @torch.no_grad()
    def _build_visual_meta(
        self,
        z_all: torch.Tensor,
        input_ids: torch.LongTensor,
        image_token_id: int,
        video_token_id: int,
        video_grid_thw: torch.LongTensor,
    ) -> BRAVisionMeta:
        image_mask = input_ids[0] == image_token_id
        n_image_tokens = image_mask.sum().item()
        frame_indices = []
        offset = 0
        spatial_merge = self.adapter.get_spatial_merge_size(self.model)
        for thw in video_grid_thw:
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            hw_llm = (h // spatial_merge) * (w // spatial_merge)
            n_tokens = t * hw_llm
            if n_image_tokens + offset + n_tokens > z_all.shape[0]:
                break
            for frame_idx in range(t):
                frame_indices.extend([frame_idx] * hw_llm)
            offset += n_tokens
        frame_tensor = torch.tensor(frame_indices, dtype=torch.long) if frame_indices else None
        return BRAVisionMeta(
            n_image_tokens=n_image_tokens,
            frame_indices=frame_tensor,
            n_visual_tokens=z_all.shape[0],
        )

    @torch.no_grad()
    def _build_temporal_diff(
        self,
        z_all: torch.Tensor,
        input_ids: torch.LongTensor,
        video_token_id: int,
        image_token_id: int,
        video_grid_thw: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        video_mask = input_ids[0] == video_token_id
        if not video_mask.any():
            return None
        image_mask = input_ids[0] == image_token_id
        n_image_tokens = image_mask.sum().item()
        z_video = z_all[n_image_tokens:]
        if z_video.shape[0] == 0:
            return None
        diffs = []
        offset = 0
        spatial_merge = self.adapter.get_spatial_merge_size(self.model)
        for thw in video_grid_thw:
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            hw_llm = (h // spatial_merge) * (w // spatial_merge)
            n_tokens = t * hw_llm
            if offset + n_tokens > z_video.shape[0]:
                break
            z_vid = z_video[offset:offset + n_tokens].view(t, hw_llm, -1)
            if t > 1:
                d = z_vid[1:] - z_vid[:-1]
                diffs.append(d.reshape(-1, d.shape[-1]))
            offset += n_tokens
        if diffs:
            return torch.cat(diffs, dim=0)
        return None

    def remove(self):
        self._hook.remove()


class BRALogitsProcessor:
    def __init__(
        self,
        extractor: BRAVisionExtractor,
        lm_head_weight: torch.Tensor,
        config: BRAConfig,
        prefill_input_ids: torch.LongTensor,
        image_token_id: int,
        video_token_id: Optional[int] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        tokenizer: Any = None,
        model: Any = None,
        adapter: Any = None,
        vasm: Optional[ProbabilisticVASM] = None,
    ):
        self.extractor = extractor
        self._model = model
        self._adapter = adapter
        self.cfg = config
        self._prefill_ids = prefill_input_ids
        self._image_token_id = image_token_id
        self._video_token_id = video_token_id
        self._video_grid_thw = video_grid_thw
        self.tokenizer = tokenizer
        self._warned_batch = False

        self.W = None if lm_head_weight.device.type == "meta" else lm_head_weight
        self._vision_features: Optional[torch.Tensor] = None
        self._temporal_diff_features: Optional[torch.Tensor] = None
        self._vision_meta = BRAVisionMeta()
        self._step = 0
        self._prev_entropy: Optional[torch.Tensor] = None
        self._stats = BRAStats()
        self._audit_log: list[dict[str, Any]] = []
        self._vasm = vasm
        self._projector = None
        self._tokenizer_family = detect_tokenizer_family(tokenizer) if tokenizer is not None else "generic"
        self._prefill_len = int(prefill_input_ids.shape[1])
        self._resolved_generated_tokens = 0
        self._pending_generation_info: Optional[dict[str, Any]] = None
        self._active_continuation: Optional[dict[str, Any]] = None
        self._stats.visual_state_provenance = dict(getattr(self.extractor, "source", {}))
        if self._adapter is not None:
            self._stats.visual_state_provenance.setdefault("adapter_name", getattr(self._adapter, "name", "unknown"))
        if self._vasm is not None:
            self._stats.vasm_metadata = self._vasm.to_dict()

    def reset(self):
        self._vision_features = None
        self._temporal_diff_features = None
        self._vision_meta = BRAVisionMeta()
        self._step = 0
        self._prev_entropy = None
        self._stats.reset()
        self._audit_log.clear()
        self._resolved_generated_tokens = 0
        self._pending_generation_info = None
        self._active_continuation = None
        self._stats.visual_state_provenance = dict(getattr(self.extractor, "source", {}))
        if self._adapter is not None:
            self._stats.visual_state_provenance.setdefault("adapter_name", getattr(self._adapter, "name", "unknown"))
        if self._vasm is not None:
            self._stats.vasm_metadata = self._vasm.to_dict()

    def get_stats(self) -> dict[str, Any]:
        return self._stats.to_dict()

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._consume_generated_token(input_ids)
        if self._vision_features is None:
            z, z_delta, meta = self.extractor.extract_vision_features(
                self._prefill_ids,
                self._image_token_id,
                self._video_token_id,
                self._video_grid_thw,
                build_temporal_diff=self.cfg.use_temporal_diff_branch,
            )
            if z is None:
                return scores
            self._vision_features = z
            self._temporal_diff_features = z_delta
            self._vision_meta = meta

        self._step += 1
        self._stats.decode_steps += 1
        if self._step <= self.cfg.warmup_steps:
            return scores

        self._stats.eligible_steps += 1
        return self._apply_bra(scores)

    def _consume_generated_token(self, input_ids: torch.LongTensor) -> None:
        generated_len = max(int(input_ids.shape[1]) - self._prefill_len, 0)
        if generated_len <= self._resolved_generated_tokens:
            return

        token_id = int(input_ids[0, -1].item())
        token = self._id_to_token(token_id)
        normalized = normalize_token(token)

        if self._active_continuation is not None:
            self._stats.continuation_attempts += 1
            if is_continuation_subword(token, self._tokenizer_family):
                self._stats.continuation_successes += 1
            else:
                self._stats.suffix_collapse_failures += 1
                if len(self._stats.continuation_failure_examples) < 10:
                    self._stats.continuation_failure_examples.append(
                        {
                            "root_step": self._active_continuation["root_step"],
                            "root_token": self._active_continuation["root_token"],
                            "next_token": token,
                            "next_token_normalized": normalized,
                            "failure_type": "suffix_collapse",
                        }
                    )
            self._active_continuation = None

        if self._pending_generation_info and self._pending_generation_info.get("intervened"):
            if self._is_prefix_trigger_token(token):
                self._active_continuation = {
                    "root_step": self._pending_generation_info["step"],
                    "root_token": token,
                    "root_token_id": token_id,
                }
        self._pending_generation_info = None
        self._resolved_generated_tokens = generated_len

    def _id_to_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            return str(token_id)
        try:
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            token = self.tokenizer.convert_ids_to_tokens(token_id)
        return token or str(token_id)

    def _is_prefix_trigger_token(self, token: str) -> bool:
        normalized = normalize_token(token)
        if not normalized or len(normalized) < 2:
            return False
        if is_continuation_subword(token, self._tokenizer_family):
            return False
        return any(ch.isalpha() for ch in normalized)

    def _get_lm_head_weight(self, device) -> torch.Tensor:
        if self.W is not None:
            w = self.W.detach()
            if w.device != device:
                w = w.to(device)
            return w
        if self._adapter is not None and self._model is not None:
            w = self._adapter.get_lm_head_weight(self._model)
            if w.device.type != "meta":
                self.W = w
                return w.detach().to(device)
        raise RuntimeError(
            "lm_head.weight is on meta device. Load the model with model.to('cuda') "
            "or a non-meta device map."
        )

    def _ensure_projector(self, input_dim: int, output_dim: int, device: torch.device) -> None:
        if self._projector is not None:
            return
        spec = ProjectorSpec(
            kind=self.cfg.projector_kind,
            checkpoint_path=self.cfg.projector_checkpoint,
        )
        self._projector = create_projector(
            spec,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            dtype=torch.float32,
        )

    def _ensure_vasm(self) -> Optional[ProbabilisticVASM]:
        if self.cfg.mask_variant == "no_mask":
            return None
        if self._vasm is not None:
            return self._vasm
        if self.cfg.vasm_artifact_path:
            self._vasm = ProbabilisticVASM.from_file(self.cfg.vasm_artifact_path)
        elif self.tokenizer is not None:
            self._vasm = ProbabilisticVASM.from_tokenizer(self.tokenizer)
        if self._vasm is not None:
            self._stats.vasm_metadata = self._vasm.to_dict()
        return self._vasm

    @torch.no_grad()
    def _apply_bra(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        cfg = self.cfg
        device = scores.device

        if scores.dim() != 2 or scores.shape[0] != 1:
            if not self._warned_batch:
                logger.warning("BRA expects batch_size=1 during generate(); got %s", tuple(scores.shape))
                self._warned_batch = True
            return scores

        candidate_k = min(cfg.top_k, scores.shape[-1])
        top_vals, top_ids = scores.topk(candidate_k, dim=-1)
        self._stats.candidate_window_sum += candidate_k

        vocab_w = self._get_lm_head_weight(device)
        w_c = vocab_w[top_ids[0]]

        entropy = self._shannon_entropy(scores)
        top_margin = float("inf")
        if top_vals.shape[-1] > 1:
            top_margin = (top_vals[0, 0] - top_vals[0, 1]).item()

        start = time.perf_counter()
        raw_scores, visual_topk, diag = self._compute_resonance(w_c)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._stats.visual_topk_sum += visual_topk
        self._stats.resonance_time_ms_sum += elapsed_ms
        self._stats.routing_time_ms_sum += elapsed_ms

        rel_scores = self._candidate_relative_scores(raw_scores)
        mask_start = time.perf_counter()
        mask = self._candidate_mask(top_ids[0], entropy)
        self._stats.vasm_time_ms_sum += (time.perf_counter() - mask_start) * 1000.0

        should_trigger, reasons = self._should_intervene(entropy, top_margin, raw_scores)
        self._prev_entropy = entropy

        if not should_trigger:
            self._record_audit(top_ids[0], raw_scores, rel_scores, mask, reasons, elapsed_ms, visual_topk, False, diag)
            self._pending_generation_info = {"step": self._step, "intervened": False}
            return scores

        self._stats.intervention_steps += 1
        l_orig = top_vals[0]
        if cfg.mode in {"legacy_entropy", "bra_v1", "bra_v1_like", "legacy_v2"}:
            rel_for_legacy = self._legacy_relative_scores(raw_scores)
            scale = l_orig.abs() if cfg.legacy_use_abs_logit_scale else torch.ones_like(l_orig)
            reward = torch.clamp(mask, min=0.25)
            l_final = l_orig - cfg.alpha * mask * (1.0 - rel_for_legacy) * scale
            l_final = l_final + cfg.beta * reward * rel_for_legacy * scale
        else:
            penalty = cfg.alpha * mask * (1.0 - rel_scores)
            l_final = l_orig - penalty

        modified = scores.clone()
        modified[0].scatter_(0, top_ids[0], l_final)

        if cfg.debug:
            self._debug_topology(top_ids[0], raw_scores, rel_scores, mask, l_orig, l_final)

        self._record_audit(top_ids[0], raw_scores, rel_scores, mask, reasons, elapsed_ms, visual_topk, True, diag)
        self._pending_generation_info = {"step": self._step, "intervened": True}
        return modified

    def _should_intervene(self, entropy: torch.Tensor, top_margin: float, raw_scores: torch.Tensor) -> tuple[bool, list[str]]:
        cfg = self.cfg
        reasons = []
        if cfg.mode in {
            "bra_zero", "bra_calib", "ablation_meanpool", "ablation_maxpool",
            "tlra_zero", "tlra_calib", "tlra_full", "tlra_adaptivetopk",
            "tlra_meanpool", "tlra_max", "tlra_randomk", "tlra_no_vasm",
        } and not (
            cfg.apply_entropy_trigger or cfg.apply_margin_trigger or cfg.apply_evidence_trigger
        ):
            reasons.append("always_after_warmup")
            return True, reasons

        trigger_terms = []
        if cfg.apply_margin_trigger:
            flag = top_margin < cfg.margin_epsilon
            trigger_terms.append(flag)
            if flag:
                reasons.append("low_margin")
        if cfg.apply_evidence_trigger:
            top_deficit = raw_scores.max().item() - raw_scores[0].item()
            flag = raw_scores[0].item() < cfg.resonance_floor and top_deficit > cfg.resonance_gap
            trigger_terms.append(flag)
            if flag:
                reasons.append("evidence_deficit")
        should = all(trigger_terms) if (trigger_terms and cfg.require_joint_trigger) else any(trigger_terms)
        if cfg.apply_entropy_trigger and self._prev_entropy is not None:
            delta_e = (entropy - self._prev_entropy).abs()
            flag = (delta_e > cfg.epsilon).any().item()
            should = should or flag
            if flag:
                reasons.append("entropy_delta")
        return should, reasons

    @torch.no_grad()
    def _compute_resonance(self, w_c: torch.Tensor) -> tuple[torch.Tensor, int, dict[str, Any]]:
        z = self._vision_features.to(w_c.device).float()
        self._ensure_projector(z.shape[-1], w_c.shape[-1], w_c.device)
        z = self._projector(z)

        w_norm = F.normalize(w_c.float(), dim=-1)
        z_norm = F.normalize(z.float(), dim=-1)

        sim = w_norm @ z_norm.T
        sim = sim / max(self.cfg.tau_sim, 1e-6)
        aggregation_mode = self._aggregation_mode()
        visual_topk = self._resolve_visual_topk(sim.shape[-1])
        static_scores, top_indices = self._aggregate_similarity(sim, aggregation_mode, visual_topk)
        final_scores = self.cfg.lambda_static * static_scores

        if self.cfg.use_temporal_diff_branch and self._temporal_diff_features is not None and self.cfg.lambda_temporal > 0:
            z_d = self._temporal_diff_features.to(w_c.device).float()
            z_d = self._projector(z_d)
            z_d_norm = F.normalize(z_d.float(), dim=-1)
            sim_t = (w_norm @ z_d_norm.T) / max(self.cfg.tau_sim, 1e-6)
            temporal_scores, _ = self._aggregate_similarity(sim_t, aggregation_mode, min(visual_topk, sim_t.shape[-1]))
            final_scores = final_scores + self.cfg.lambda_temporal * temporal_scores

        diag = {"top_indices": top_indices[:1].detach().cpu() if top_indices is not None else None}
        return final_scores, visual_topk, diag

    def _aggregation_mode(self) -> str:
        if self.cfg.aggregation_mode:
            return self.cfg.aggregation_mode
        mode = self.cfg.mode.lower()
        if mode == "ablation_meanpool":
            return "mean"
        if mode == "ablation_maxpool":
            return "max"
        if mode == "tlra_meanpool":
            return "mean"
        if mode == "tlra_randomk":
            return "random_k"
        return "adaptive_topk"

    def _resolve_visual_topk(self, n_visual_tokens: int) -> int:
        if n_visual_tokens <= 0:
            return 1
        k = max(self.cfg.visual_topk_min, math.ceil(self.cfg.rho * n_visual_tokens))
        k = min(k, self.cfg.visual_topk_max, n_visual_tokens)
        return max(k, 1)

    def _aggregate_similarity(
        self,
        sim: torch.Tensor,
        aggregation_mode: str,
        visual_topk: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if aggregation_mode == "mean":
            return sim.mean(dim=-1), None
        if aggregation_mode == "max":
            values, indices = sim.max(dim=-1)
            return values, indices.unsqueeze(-1)
        if aggregation_mode == "random_k":
            if sim.shape[-1] <= visual_topk:
                indices = torch.arange(sim.shape[-1], device=sim.device).expand(sim.shape[0], -1)
            else:
                generator = torch.Generator(device=sim.device)
                generator.manual_seed(int(self.cfg.random_patch_seed + self._step))
                sampled = torch.rand(sim.shape, device=sim.device, generator=generator)
                _, indices = sampled.topk(visual_topk, dim=-1)
            values = sim.gather(-1, indices)
            return values.mean(dim=-1), indices
        values, indices = sim.topk(visual_topk, dim=-1)
        return values.mean(dim=-1), indices

    def _candidate_relative_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(raw_scores / max(self.cfg.candidate_temperature, 1e-6), dim=-1)

    def _legacy_relative_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        lo = raw_scores.min()
        hi = raw_scores.max()
        if (hi - lo).abs().item() < 1e-6:
            return raw_scores.clamp(0.0, 1.0)
        minmax = ((raw_scores - lo) / (hi - lo)).clamp(0.0, 1.0)
        mix = self.cfg.legacy_relative_mix
        return (1.0 - mix) * raw_scores.sigmoid() + mix * minmax

    def _candidate_mask(self, token_ids: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        variant = self.cfg.mask_variant.lower()
        if variant == "no_mask":
            return torch.ones_like(token_ids, dtype=torch.float32, device=token_ids.device)
        if variant == "entropy_mask":
            vocab_size = max(self._get_lm_head_weight(token_ids.device).shape[0], 2)
            entropy_norm = (entropy.float() / math.log(vocab_size)).clamp(0.0, 1.0)
            return torch.full_like(token_ids, float(entropy_norm.item()), dtype=torch.float32, device=token_ids.device)

        vasm = self._ensure_vasm()
        if vasm is None:
            gamma = torch.ones_like(token_ids, dtype=torch.float32, device=token_ids.device)
        else:
            gamma = vasm.lookup(token_ids, device=token_ids.device).float()
        if variant == "binary_mask":
            gamma = (gamma > 0.5).float()
        return gamma

    @staticmethod
    def _shannon_entropy(logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits.float(), dim=-1)
        return -(p * (p + 1e-12).log()).sum(dim=-1)

    def _record_audit(
        self,
        top_ids: torch.Tensor,
        raw_scores: torch.Tensor,
        rel_scores: torch.Tensor,
        mask: torch.Tensor,
        reasons: list[str],
        elapsed_ms: float,
        visual_topk: int,
        intervened: bool,
        diag: dict[str, Any],
    ) -> None:
        if len(self._audit_log) < self.cfg.audit_max_steps:
            self._audit_log.append(
                {
                    "step": self._step,
                    "token_ids": top_ids.tolist()[:10],
                    "raw_scores": [round(float(x), 6) for x in raw_scores[:10].tolist()],
                    "rel_scores": [round(float(x), 6) for x in rel_scores[:10].tolist()],
                    "mask": [round(float(x), 6) for x in mask[:10].tolist()],
                    "reasons": reasons,
                    "intervened": intervened,
                    "resonance_time_ms": round(elapsed_ms, 4),
                    "routing_time_ms": round(elapsed_ms, 4),
                    "visual_topk": int(visual_topk),
                }
            )

        if diag.get("top_indices") is not None and self._vision_meta.frame_indices is not None:
            frame_tensor = self._vision_meta.frame_indices
            for idx in diag["top_indices"][0].tolist():
                rel_idx = idx - self._vision_meta.n_image_tokens
                if 0 <= rel_idx < len(frame_tensor):
                    frame_idx = int(frame_tensor[rel_idx].item())
                    self._stats.selected_frame_histogram[frame_idx] = (
                        self._stats.selected_frame_histogram.get(frame_idx, 0) + 1
                    )

    def _debug_topology(
        self,
        top_ids: torch.Tensor,
        raw_scores: torch.Tensor,
        rel_scores: torch.Tensor,
        mask: torch.Tensor,
        l_orig: torch.Tensor,
        l_final: torch.Tensor,
    ) -> None:
        tokens = None
        if self.tokenizer is not None:
            tokens = self.tokenizer.convert_ids_to_tokens(top_ids.tolist())
        for i in range(min(10, top_ids.shape[0])):
            tok = tokens[i] if tokens is not None else str(int(top_ids[i].item()))
            print(
                f"  [{i:02d}] {tok:20s} raw={float(raw_scores[i]):+.4f} "
                f"rel={float(rel_scores[i]):.4f} mask={float(mask[i]):.2f} "
                f"logit={float(l_orig[i]):+.4f}->{float(l_final[i]):+.4f}"
            )


def create_bra_processor(
    model,
    adapter,
    prefill_input_ids,
    config=None,
    tokenizer=None,
    video_grid_thw=None,
):
    cfg = config or BRAConfig()
    extractor = BRAVisionExtractor(model, adapter)
    image_token_id = adapter.get_image_token_id(model)
    video_token_id = adapter.get_video_token_id(model)
    lm_head_weight = adapter.get_lm_head_weight(model)
    vasm = None
    if cfg.mask_variant != "no_mask" and tokenizer is not None:
        if cfg.vasm_artifact_path:
            vasm = ProbabilisticVASM.from_file(cfg.vasm_artifact_path)
        else:
            vasm = ProbabilisticVASM.from_tokenizer(tokenizer)

    processor = BRALogitsProcessor(
        extractor=extractor,
        lm_head_weight=lm_head_weight,
        config=cfg,
        prefill_input_ids=prefill_input_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        video_grid_thw=video_grid_thw,
        tokenizer=tokenizer,
        model=model,
        adapter=adapter,
        vasm=vasm,
    )
    return extractor, processor


def make_bra_config(mode: str = "bra_zero", **overrides) -> BRAConfig:
    mode = (mode or "bra_zero").lower()
    if mode in {"bra_zero", "zero", "bra", "v8", "bra_v2"}:
        cfg = BRAConfig(mode="bra_zero")
    elif mode in {"tlra", "tlra_zero"}:
        cfg = BRAConfig(mode="tlra_zero")
    elif mode in {"tlra_full", "tlra_adaptivetopk"}:
        cfg = BRAConfig(mode=mode, projector_kind="loaded_linear")
    elif mode == "tlra_randomk":
        cfg = BRAConfig(mode="tlra_randomk", projector_kind="loaded_linear", aggregation_mode="random_k")
    elif mode in {"bra_calib", "calib"}:
        cfg = BRAConfig(mode="bra_calib", projector_kind="loaded_linear")
    elif mode == "tlra_calib":
        cfg = BRAConfig(mode="tlra_calib", projector_kind="loaded_linear")
    elif mode in {"ablation_meanpool", "meanpool", "bra_meanpool"}:
        cfg = BRAConfig(mode="ablation_meanpool")
    elif mode == "tlra_meanpool":
        cfg = BRAConfig(mode="tlra_meanpool", projector_kind="loaded_linear", aggregation_mode="mean")
    elif mode in {"ablation_maxpool", "maxpool", "bra_maxpool", "bra_max"}:
        cfg = BRAConfig(mode="ablation_maxpool")
    elif mode == "tlra_max":
        cfg = BRAConfig(mode="tlra_max", projector_kind="loaded_linear", aggregation_mode="max")
    elif mode in {"bra_no_vasm", "no_vasm"}:
        cfg = BRAConfig(mode="bra_no_vasm", mask_variant="no_mask")
    elif mode == "tlra_no_vasm":
        cfg = BRAConfig(mode="tlra_no_vasm", projector_kind="loaded_linear", mask_variant="no_mask")
    elif mode in {"legacy_entropy", "bra_v1", "v1", "bra_v1_like", "tlra_v1_like"}:
        cfg = BRAConfig(
            mode="tlra_v1_like" if mode == "tlra_v1_like" else "bra_v1",
            alpha=0.20,
            beta=0.15,
            apply_entropy_trigger=True,
            apply_margin_trigger=False,
            apply_evidence_trigger=False,
            require_joint_trigger=False,
            mask_variant="binary_mask",
        )
    elif mode in {"legacy_v2"}:
        cfg = BRAConfig(
            mode="legacy_v2",
            alpha=0.20,
            beta=0.15,
            apply_entropy_trigger=True,
            apply_margin_trigger=True,
            apply_evidence_trigger=True,
            require_joint_trigger=True,
            mask_variant="vasm",
        )
    else:
        raise ValueError(f"Unknown BRA mode: {mode}")

    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
