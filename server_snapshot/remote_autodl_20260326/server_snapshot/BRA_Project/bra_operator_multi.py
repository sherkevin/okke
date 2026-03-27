"""
Compatibility shim for legacy BRA operator usage.

The adapter abstractions stay here, but the BRA mathematics now lives in
`bra_logits_processor.py`. This module only injects the canonical
`BRALogitsProcessor` into `model.generate(...)`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from bra_logits_processor import BRAConfig, create_bra_processor

logger = logging.getLogger(__name__)


class ModelAdapter:
    name: str = "unknown"

    def get_inner_model(self, model):
        raise NotImplementedError

    def get_lm_head_weight(self, model):
        raise NotImplementedError

    def get_image_token_id(self, model) -> int:
        raise NotImplementedError

    def get_video_token_id(self, model) -> Optional[int]:
        return None

    def supports_video(self) -> bool:
        return False

    def get_spatial_merge_size(self, model) -> int:
        return 1

    def has_pixel_values(self, kwargs) -> bool:
        return kwargs.get("pixel_values") is not None

    def describe_visual_state_source(self, model) -> dict[str, Any]:
        inner = self.get_inner_model(model)
        return {
            "module_path": type(inner).__name__,
            "capture_kind": "forward_output_last_hidden_state",
            "layer_role": "final",
            "state_family": "llm_hidden_state",
        }


class Qwen3VLAdapter(ModelAdapter):
    name = "qwen3_vl"

    def get_inner_model(self, model):
        return model.model

    def get_lm_head_weight(self, model):
        return model.lm_head.weight

    def get_image_token_id(self, model) -> int:
        return model.config.image_token_id

    def get_video_token_id(self, model) -> Optional[int]:
        return model.config.video_token_id

    def supports_video(self) -> bool:
        return True

    def get_spatial_merge_size(self, model) -> int:
        return model.config.vision_config.spatial_merge_size

    def has_pixel_values(self, kwargs) -> bool:
        return (kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None)

    def describe_visual_state_source(self, model) -> dict[str, Any]:
        return {
            "module_path": "model.model",
            "capture_kind": "forward_output_last_hidden_state",
            "layer_role": "final",
            "state_family": "llm_hidden_state",
            "source_note": "h_L^(v_j) captured from final Qwen language model hidden states at visual token positions during prefill.",
        }


class LLaVAAdapter(ModelAdapter):
    name = "llava"

    def get_inner_model(self, model):
        if hasattr(model, "language_model"):
            return model.language_model.model
        return model.model

    def get_lm_head_weight(self, model):
        if hasattr(model, "language_model"):
            return model.language_model.lm_head.weight
        return model.lm_head.weight

    def get_image_token_id(self, model) -> int:
        return getattr(model.config, "image_token_index", 32000)

    def describe_visual_state_source(self, model) -> dict[str, Any]:
        module_path = "language_model.model" if hasattr(model, "language_model") else "model"
        return {
            "module_path": module_path,
            "capture_kind": "forward_output_last_hidden_state",
            "layer_role": "final",
            "state_family": "llm_hidden_state",
            "source_note": "h_L^(v_j) captured from final LLaVA language model hidden states at image token positions during prefill.",
        }


class InstructBLIPAdapter(ModelAdapter):
    name = "instructblip"

    def get_inner_model(self, model):
        if hasattr(model, "language_model"):
            return model.language_model.model
        return model.model

    def get_lm_head_weight(self, model):
        if hasattr(model, "language_model"):
            return model.language_model.lm_head.weight
        return model.lm_head.weight

    def get_image_token_id(self, model) -> int:
        return getattr(model.config, "image_token_index", 32001)

    def describe_visual_state_source(self, model) -> dict[str, Any]:
        module_path = "language_model.model" if hasattr(model, "language_model") else "model"
        return {
            "module_path": module_path,
            "capture_kind": "forward_output_last_hidden_state",
            "layer_role": "final",
            "state_family": "llm_hidden_state",
            "source_note": "h_L^(v_j) captured from final InstructBLIP language model hidden states at query/image token positions during prefill.",
        }


def detect_adapter(model) -> ModelAdapter:
    cls_name = type(model).__name__.lower()
    model_type = getattr(model.config, "model_type", "").lower()
    if "qwen3" in cls_name or "qwen3" in model_type or "qwen2" in cls_name or "qwen2" in model_type:
        return Qwen3VLAdapter()
    if "llava" in cls_name or "llava" in model_type:
        return LLaVAAdapter()
    if "instructblip" in cls_name or "instructblip" in model_type:
        return InstructBLIPAdapter()
    raise ValueError(
        f"Unsupported model: {type(model).__name__} (model_type={model_type}). "
        f"Supported: Qwen3-VL, LLaVA-1.5, InstructBLIP"
    )


class BRAOperatorMulti:
    """
    Backward-compatible wrapper that injects the canonical logits processor into
    `model.generate(...)`. It preserves the old `BRAOperator`-style API without
    keeping a separate math implementation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: BRAConfig | None = None,
        tokenizer: Any | None = None,
        adapter: ModelAdapter | None = None,
    ):
        self.model = model
        self.cfg = config or BRAConfig()
        self.tokenizer = tokenizer
        self.adapter = adapter or detect_adapter(model)
        self._original_generate = model.generate
        self._current = None
        self.model.generate = self._wrapped_generate
        logger.info("BRA compatibility shim attached to %s", self.adapter.name)

    def _wrapped_generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        if input_ids is None:
            return self._original_generate(*args, **kwargs)

        tokenizer = self.tokenizer
        if tokenizer is None and hasattr(self, "processor"):
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
        video_grid_thw = kwargs.get("video_grid_thw")
        extractor, bra_proc = create_bra_processor(
            self.model,
            self.adapter,
            input_ids,
            config=self.cfg,
            tokenizer=tokenizer,
            video_grid_thw=video_grid_thw,
        )
        self._current = (extractor, bra_proc)

        logits_processor = list(kwargs.pop("logits_processor", []) or [])
        logits_processor.append(bra_proc)
        kwargs["logits_processor"] = logits_processor
        try:
            return self._original_generate(*args, **kwargs)
        finally:
            extractor.remove()
            bra_proc.reset()
            self._current = None

    def reset(self) -> None:
        if self._current is not None:
            _, bra_proc = self._current
            bra_proc.reset()

    def remove(self) -> None:
        if self._current is not None:
            extractor, bra_proc = self._current
            extractor.remove()
            bra_proc.reset()
            self._current = None
        self.model.generate = self._original_generate


def create_bra_operator(model, config=None, tokenizer=None, adapter=None):
    return BRAOperatorMulti(model, config=config, tokenizer=tokenizer, adapter=adapter)
