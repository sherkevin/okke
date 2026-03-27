#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny POPE parity smoke on the remote CHIRO tree.")
    parser.add_argument("--remote-root", default="/root/autodl-tmp/CHIRO")
    parser.add_argument("--model", default="llava-1.5")
    parser.add_argument("--pope-type", default="random")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=50.0)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num-attn-candidates", type=int, default=5)
    parser.add_argument("--penalty-weights", type=float, default=1.0)
    parser.add_argument("--data-path", default="/root/autodl-tmp/BRA_Project/datasets/coco2014/val2014/")
    parser.add_argument("--llava-ckpt", default="/root/autodl-tmp/BRA_Project/models/llava-1.5-7b-hf")
    parser.add_argument("--llava-proc-path", default="/root/chiro_assets/clip-vit-large-patch14-336")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--output-json")
    return parser.parse_args()


def compute_metrics(preds: list[int], labels: list[int]) -> dict[str, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    yes_ratio = sum(preds) / max(1, len(preds))
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "yes_ratio": round(yes_ratio, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def prepare_llava_eval_config(remote_root: Path, llava_ckpt: str, llava_proc_path: str) -> Path:
    source = remote_root / "eval_configs" / "llava-1.5_eval.yaml"
    text = source.read_text(encoding="utf-8")
    text = re.sub(r'merged_ckpt:\s*".*?"', f'merged_ckpt: "{llava_ckpt}"', text)
    target = remote_root / "tests" / "_tmp_llava_eval.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def main() -> int:
    args = parse_args()
    remote_root = Path(args.remote_root).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(remote_root))
    sys.path.insert(0, str(remote_root / "transformers-4.29.2" / "src"))
    os.chdir(remote_root)

    import torch
    import pope_eval
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.models import load_preprocess

    if args.model != "llava-1.5":
        raise ValueError("The first parity harness only supports llava-1.5.")

    print("Preparing eval config...", flush=True)
    cfg_path = prepare_llava_eval_config(remote_root, args.llava_ckpt, args.llava_proc_path)

    runtime_args = argparse.Namespace(
        model=args.model,
        pope_type=args.pope_type,
        gpu_id=args.gpu_id,
        options=None,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        beam=args.beam,
        sample=args.sample,
        scale_factor=args.scale_factor,
        threshold=args.threshold,
        num_attn_candidates=args.num_attn_candidates,
        penalty_weights=args.penalty_weights,
        cfg_path=str(cfg_path),
        pope_path=pope_eval.POPE_PATH[args.pope_type],
    )

    print("Building config...", flush=True)
    cfg = Config(runtime_args)
    cfg.get_config().preprocess.vis_processor.train.proc_type = args.llava_proc_path
    cfg.get_config().preprocess.vis_processor.eval.proc_type = args.llava_proc_path
    pope_eval.setup_seeds(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...", flush=True)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    print("Loading preprocessors...", flush=True)
    vis_processors, _txt_processors = load_preprocess(cfg.get_config().preprocess)

    print("Building dataset...", flush=True)
    dataset = pope_eval.POPEDataSet(
        pope_path=runtime_args.pope_path,
        data_path=runtime_args.data_path,
        trans=vis_processors["eval"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=runtime_args.batch_size,
        shuffle=False,
        num_workers=runtime_args.num_workers,
        drop_last=False,
    )

    template = pope_eval.INSTRUCTION_TEMPLATE[args.model]
    outputs: list[dict[str, object]] = []
    pred_list: list[int] = []
    label_list: list[int] = []

    print("Running generation...", flush=True)
    print(
        f"Using official CHIRO OPERA path: opera_decoding=True, beam={args.beam}, "
        f"num_attn_candidates={args.num_attn_candidates}, max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    for batch_idx, data in enumerate(loader):
        if batch_idx >= args.n_samples:
            break
        image = data["image"].to(device)
        query = data["query"]
        label = [int(x) for x in data["label"]]
        prompts = [template.replace("<question>", q) for q in query]
        with torch.inference_mode():
            out = model.generate(
                {"image": image, "prompt": prompts},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )
        pred_list = pope_eval.recorder(out, pred_list)
        label_list.extend(label)
        outputs.append(
            {
                "batch_idx": batch_idx,
                "query": query,
                "label": label,
                "answer": out,
            }
        )
        print(f"Finished batch {batch_idx}", flush=True)

    payload = {
        "model": args.model,
        "pope_type": args.pope_type,
        "n_samples": len(outputs),
        "settings": {
            "opera_decoding": True,
            "beam": args.beam,
            "scale_factor": args.scale_factor,
            "threshold": args.threshold,
            "num_attn_candidates": args.num_attn_candidates,
            "penalty_weights": args.penalty_weights,
            "max_new_tokens": args.max_new_tokens,
            "gpu_id": args.gpu_id,
        },
        "metrics": compute_metrics(pred_list, label_list),
        "outputs": outputs,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print("Completed parity smoke.", flush=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
