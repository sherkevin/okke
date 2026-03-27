#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file


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
    parser = argparse.ArgumentParser(description="Convert a HF LLaVA checkpoint into an official-OPERA-compatible checkpoint.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--projector-type", default="mlp2x_gelu")
    parser.add_argument("--vision-select-feature", default="patch")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_official_config(source: Path, clip_path: str, projector_type: str, vision_select_feature: str) -> dict:
    raw_cfg = load_json(source / "config.json")
    text_cfg = dict(raw_cfg["text_config"])
    text_cfg["model_type"] = "llava"
    text_cfg["architectures"] = ["LlavaLlamaForCausalLM"]
    text_cfg["mm_vision_tower"] = clip_path
    text_cfg["mm_hidden_size"] = raw_cfg["vision_config"]["hidden_size"]
    text_cfg["mm_projector_type"] = projector_type
    text_cfg["mm_vision_select_layer"] = raw_cfg.get("vision_feature_layer", -2)
    text_cfg["mm_vision_select_feature"] = vision_select_feature
    text_cfg["mm_use_im_start_end"] = True
    text_cfg["mm_use_im_patch_token"] = True
    text_cfg["use_mm_proj"] = True
    return text_cfg


def map_hf_weight_key(key: str) -> str | None:
    if key.startswith("language_model.model."):
        return "model." + key[len("language_model.model.") :]
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head.") :]
    if key.startswith("multi_modal_projector.linear_1."):
        return "model.mm_projector.0." + key[len("multi_modal_projector.linear_1.") :]
    if key.startswith("multi_modal_projector.linear_2."):
        return "model.mm_projector.2." + key[len("multi_modal_projector.linear_2.") :]
    return None


def hf_shard_paths(source: Path) -> list[Path]:
    index = load_json(source / "model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    return [source / name for name in shard_names]


def convert_state_dict(source: Path) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for shard_path in hf_shard_paths(source):
        tensors = load_file(str(shard_path))
        for key, value in tensors.items():
            new_key = map_hf_weight_key(key)
            if new_key is not None:
                converted[new_key] = value
        del tensors
    if not converted:
        raise RuntimeError("No tensors were converted from the HF checkpoint.")
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

    config = build_official_config(source, args.clip_path, args.projector_type, args.vision_select_feature)
    converted_state = convert_state_dict(source)

    (target / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(converted_state, target / "pytorch_model.bin", _use_new_zipfile_serialization=False)
    copy_support_files(source, target)

    print(
        json.dumps(
            {
                "target": str(target),
                "n_tensors": len(converted_state),
                "support_files": [name for name in COPY_FILES if (target / name).exists()],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
