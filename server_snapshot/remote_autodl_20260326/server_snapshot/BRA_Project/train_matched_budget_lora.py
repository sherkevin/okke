#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="Matched-budget Base+LoRA scaffold for Phi_calib data.")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-jsonl", default=None, help="JSONL records with prompt/answer and optional image_path.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-key", default="qwen3vl2b")
    parser.add_argument("--data-tag", default="phi_calib_data")
    parser.add_argument("--objective-tag", default="ce")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sanity-check-load", action="store_true")
    parser.add_argument("--train-records-target", type=int, default=50000)
    parser.add_argument("--matched-budget-ref", default="Phi_calib")
    parser.add_argument("--data-budget-note", default="50k conceptual captions, intended to match Phi_calib supervision budget")
    parser.add_argument("--training-budget-note", default="Budget-matched LoRA control with fixed seed and identical optimizer schedule family")
    parser.add_argument("--objective-note", default="VASM-masked next-token cross-entropy intended to match Phi_calib target function")
    args = parser.parse_args()
    if args.config_json:
        config_payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        for key, value in config_payload.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)
    for field in ("model_path", "train_jsonl", "output_root"):
        if not getattr(args, field):
            raise SystemExit(f"Missing required argument: --{field.replace('_', '-')}")
    return args


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str | Path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError("Training JSONL is empty.")
    return records


def build_run_name(args) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"base_lora_{args.model_key}"
        f"_data-{args.data_tag}"
        f"_obj-{args.objective_tag}"
        f"_steps-{args.max_steps}"
        f"_r-{args.lora_r}"
        f"_a-{args.lora_alpha}"
        f"_{ts}"
    )


def build_messages(record: dict):
    user_content = []
    if record.get("image_path"):
        user_content.append({"type": "image", "image": record["image_path"]})
    user_content.append({"type": "text", "text": record["prompt"]})
    messages = [{"role": "user", "content": user_content}]
    if record.get("answer"):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": record["answer"]}]})
    return messages


def build_inputs(processor, record: dict, device: torch.device):
    messages = build_messages(record)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    skip = {"mm_token_type_ids"}
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items() if k not in skip}
    labels = inputs["input_ids"].clone()
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) if hasattr(processor, "tokenizer") else None
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def main():
    args = parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)
    print(
        "training_config",
        json.dumps(
            {
                "seed": args.seed,
                "data_tag": args.data_tag,
                "objective_tag": args.objective_tag,
                "max_steps": args.max_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            ensure_ascii=False,
        ),
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()

    records = load_records(args.train_jsonl)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = build_run_name(args)
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    for step in range(args.max_steps):
        record = records[step % len(records)]
        inputs = build_inputs(processor, record, device)
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"step={step + 1} loss={loss_value:.6f}")
        if args.save_every and (step + 1) % args.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-step-{step + 1}"
            model.save_pretrained(ckpt_dir)

    model.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir / "processor")

    meta = {
        "run_name": run_name,
        "seed": args.seed,
        "model_path": args.model_path,
        "train_jsonl": str(Path(args.train_jsonl).resolve()),
        "data_tag": args.data_tag,
        "objective_tag": args.objective_tag,
        "train_records_target": args.train_records_target,
        "loaded_record_count": len(records),
        "matched_budget_ref": args.matched_budget_ref,
        "data_budget_note": args.data_budget_note,
        "training_budget_note": args.training_budget_note,
        "objective_note": args.objective_note,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "device": str(device),
        "loss_history": losses,
        "checkpoint_naming_rule": "base_lora_{model_key}_data-{data_tag}_obj-{objective_tag}_steps-{max_steps}_r-{lora_r}_a-{lora_alpha}_{timestamp}",
    }
    if args.sanity_check_load:
        from peft import PeftModel

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        loaded = PeftModel.from_pretrained(base_model, output_dir)
        meta["sanity_check"] = {
            "checkpoint_loadable": True,
            "loaded_adapter_class": loaded.__class__.__name__,
        }
        print(f"sanity_check_load=ok adapter_class={loaded.__class__.__name__}")
    (output_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LoRA scaffold checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
