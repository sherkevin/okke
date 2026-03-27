#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal load validation for a LoRA checkpoint.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    adapter_path = Path(args.adapter_path)
    meta_path = adapter_path / "training_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base_model).to(args.device)
    processor = AutoProcessor.from_pretrained(args.base_model)
    loaded = PeftModel.from_pretrained(model, adapter_path)

    payload = {
        "ok": True,
        "adapter_path": str(adapter_path),
        "base_model": args.base_model,
        "loaded_adapter_class": loaded.__class__.__name__,
        "processor_loaded": processor is not None,
        "training_meta_present": meta is not None,
        "seed": None if meta is None else meta.get("seed"),
        "objective_tag": None if meta is None else meta.get("objective_tag"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
