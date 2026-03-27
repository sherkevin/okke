#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import LlavaForConditionalGeneration


COPY_FILES = [
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a HF Llava checkpoint into a CHIRO-compatible checkpoint.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--projector-type", default="mlp2x_gelu")
    parser.add_argument("--vision-select-feature", default="patch")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_chiro_config(model: LlavaForConditionalGeneration, clip_path: str, projector_type: str, vision_select_feature: str) -> dict:
    cfg = model.config
    text_cfg = cfg.text_config.to_dict()
    text_cfg["model_type"] = "llava"
    text_cfg["architectures"] = ["LlavaLlamaForCausalLM"]
    text_cfg["mm_vision_tower"] = clip_path
    text_cfg["mm_hidden_size"] = cfg.vision_config.hidden_size
    text_cfg["mm_projector_type"] = projector_type
    text_cfg["mm_vision_select_layer"] = cfg.vision_feature_layer
    text_cfg["mm_vision_select_feature"] = vision_select_feature
    text_cfg["mm_use_im_start_end"] = True
    text_cfg["mm_use_im_patch_token"] = True
    text_cfg["use_mm_proj"] = True
    return text_cfg


def convert_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in hf_state_dict.items():
        if key.startswith("language_model.model."):
            converted["model." + key[len("language_model.model."):]] = value
        elif key.startswith("language_model.lm_head."):
            converted["lm_head." + key[len("language_model.lm_head."):]] = value
        elif key.startswith("multi_modal_projector.linear_1."):
            converted["model.mm_projector.0." + key[len("multi_modal_projector.linear_1."):]] = value
        elif key.startswith("multi_modal_projector.linear_2."):
            converted["model.mm_projector.2." + key[len("multi_modal_projector.linear_2."):]] = value
    return converted


def copy_support_files(source: Path, target: Path) -> None:
    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    if target.exists():
        if not args.force:
            raise SystemExit(f"Target already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    model = LlavaForConditionalGeneration.from_pretrained(str(source), low_cpu_mem_usage=True)
    config = build_chiro_config(model, args.clip_path, args.projector_type, args.vision_select_feature)
    converted_state = convert_state_dict(model.state_dict())

    (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(converted_state, target / "pytorch_model.bin", _use_new_zipfile_serialization=False)
    copy_support_files(source, target)
    print(json.dumps({"target": str(target), "n_tensors": len(converted_state)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
