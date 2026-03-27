from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CHORDConfig:
    enabled: bool = False
    alpha_anchor: float = 0.5
    lambda_cur: float = 0.25
    lambda_fut: float = 0.5
    lambda_txt: float = 1.0
    future_horizon: int = 4
    future_topk: int = 5
    detector_box_threshold: float = 0.25
    detector_text_threshold: float = 0.25
    max_boxes: int = 10
    grounding_dino_path: str = "/root/autodl-tmp/BRA_Project/models/grounding-dino-base"
    detector_python: str = "/root/miniconda3/bin/python"
    diagnostics_jsonl: str | None = None

    @classmethod
    def from_args(cls, args) -> "CHORDConfig":
        explicit_chord_fields = (
            "alpha_anchor",
            "lambda_cur",
            "lambda_fut",
            "lambda_txt",
            "future_horizon",
            "future_topk",
            "detector_box_threshold",
            "detector_text_threshold",
            "max_boxes",
        )
        enabled = bool(
            getattr(args, "chord_enable", False)
            or any(getattr(args, field, None) is not None for field in explicit_chord_fields)
        )
        return cls(
            enabled=enabled,
            alpha_anchor=getattr(args, "alpha_anchor", None) if getattr(args, "alpha_anchor", None) is not None else 0.5,
            lambda_cur=getattr(args, "lambda_cur", None) if getattr(args, "lambda_cur", None) is not None else 0.25,
            lambda_fut=getattr(args, "lambda_fut", None) if getattr(args, "lambda_fut", None) is not None else 0.5,
            lambda_txt=getattr(args, "lambda_txt", None) if getattr(args, "lambda_txt", None) is not None else 1.0,
            future_horizon=getattr(args, "future_horizon", None) if getattr(args, "future_horizon", None) is not None else 4,
            future_topk=getattr(args, "future_topk", None) if getattr(args, "future_topk", None) is not None else 5,
            detector_box_threshold=getattr(args, "detector_box_threshold", None) if getattr(args, "detector_box_threshold", None) is not None else 0.25,
            detector_text_threshold=getattr(args, "detector_text_threshold", None) if getattr(args, "detector_text_threshold", None) is not None else 0.25,
            max_boxes=getattr(args, "max_boxes", None) if getattr(args, "max_boxes", None) is not None else 10,
            grounding_dino_path=getattr(
                args,
                "grounding_dino_path",
                "/root/autodl-tmp/BRA_Project/models/grounding-dino-base",
            ),
            detector_python=getattr(args, "detector_python", "/root/miniconda3/bin/python"),
            diagnostics_jsonl=getattr(args, "chord_log_jsonl", None),
        )
