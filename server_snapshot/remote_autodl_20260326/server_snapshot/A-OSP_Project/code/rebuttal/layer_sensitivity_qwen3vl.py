"""
Layer Sensitivity Scan — Reviewer Q3 (L-4 Generalizability)
============================================================
Verifies if the "L-4" sweet spot is relatively or absolutely positioned
across Qwen3-VL-2B and Qwen3-VL-8B. Scans layers L-2 to L-10, 50 POPE samples.

Model: Qwen3-VL-2B and Qwen3-VL-8B.
Output: logs/rebuttal/layer_sensitivity_qwen3vl_2b.csv, _8b.csv

Usage:
  python layer_sensitivity_qwen3vl.py --model 2b
  python layer_sensitivity_qwen3vl.py --model 8b
  python layer_sensitivity_qwen3vl.py --both
"""

import sys, os, gc, json, csv, argparse, time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.stdout.reconfigure(line_buffering=True)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
BLUR_DIR = PROJECT / "data" / "blurred_calibration" / "blur"
COCO_DIR = PROJECT / "data" / "coco_val2014"
POPE_FILE = PROJECT / "data" / "pope" / "pope_coco_popular.jsonl"
OUT_LOGS = PROJECT / "logs" / "rebuttal"
REGISTRY_PATH = PROJECT / "DATA_REGISTRY.md"

K = 20
MAX_NEW_TOKENS_CAL = 20
MAX_NEW_TOKENS_POPE = 16
NUM_CALIBRATION = 50
POPE_LIMIT = 50

SCAN_LAYER_OFFSETS = [-2, -3, -4, -5, -6, -7, -8, -9, -10]


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def locate_entity_position(gen_ids, tokenizer):
    DETERMINERS = frozenset({"a", "an", "the"})
    TEMPLATE = frozenset({"image", "picture", "photo", "shows", "depicts"})
    decoded = [tokenizer.decode([tid], skip_special_tokens=True).strip().lower() for tid in gen_ids]
    for i, tok in enumerate(decoded):
        if tok in DETERMINERS and (i + 1) < len(decoded):
            c = decoded[i + 1]
            if len(c) > 2 and c.isalpha() and c not in TEMPLATE:
                return i + 1
    for i, tok in enumerate(decoded):
        if len(tok) > 3 and tok.isalpha() and tok not in TEMPLATE and tok not in DETERMINERS:
            return i
    return max(1, len(decoded) // 3)


class LayerCapture:
    def __init__(self):
        self.step_states = []
        self._prefill_done = False

    def reset(self):
        self.step_states.clear()
        self._prefill_done = False

    def __call__(self, module, inp, out):
        h = out[0]
        if not self._prefill_done:
            self._prefill_done = True
            return
        self.step_states.append(h[:, -1, :].detach().float().cpu())


class LayerAOSPHook:
    def __init__(self, decoder_layers, layer_idx, V_bias, L_prior, alpha=0.5, mu=0.0, burn_in=0):
        self.V = V_bias.clone()
        self.alpha, self.mu, self.L_prior_val = alpha, mu, L_prior
        self.L_bar = L_prior
        self.t = 0
        self.burn_in = burn_in
        self.interventions = 0
        self.total_steps = 0
        self._handle = decoder_layers[layer_idx].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        h = out[0]
        self.t += 1
        self.total_steps += 1
        if self.t <= self.burn_in:
            return
        H = h[:, -1, :].clone()
        V = self.V.to(H.device, dtype=H.dtype)
        proj = H @ V.T
        L_t = torch.sqrt((proj ** 2).sum(-1)).item()
        if self.mu == 0 or L_t > self.mu * self.L_bar:
            Hp = H - self.alpha * (proj @ V)
            on = torch.clamp(H.norm(-1, keepdim=True), min=1e-6)
            pn = torch.clamp(Hp.norm(-1, keepdim=True), min=1e-6)
            h[:, -1, :] = Hp / pn * on
            self.interventions += 1

    def reset(self):
        self.t = 0
        self.L_bar = self.L_prior_val
        self.interventions = 0
        self.total_steps = 0

    def remove(self):
        self._handle.remove()


def load_pope_image(img_name):
    for ext in ["", ".jpg", ".png"]:
        p = COCO_DIR / (img_name + ext)
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


@torch.no_grad()
def run_model(model_id: str, model_size: str):
    print(f"\n{'='*60}")
    print(f"Qwen3-VL-{model_size.upper()} — Layer Sensitivity Scan")
    print(f"{'='*60}")

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        print("ERROR: Qwen3VLForConditionalGeneration not found.")
        sys.exit(1)

    try:
        import flash_attn
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"

    print(f"Loading {model_id} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    tokenizer = processor.tokenizer

    if hasattr(model.model, "language_model"):
        decoder_layers = model.model.language_model.layers
    elif hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        decoder_layers = model.model.model.layers
    else:
        decoder_layers = model.model.layers
    num_layers = len(decoder_layers)
    print(f"Decoder layers: {num_layers}")

    import glob
    cal_paths = sorted(glob.glob(str(BLUR_DIR / "*.jpg")))[:NUM_CALIBRATION]
    if not cal_paths:
        cal_paths = sorted(glob.glob(str(BLUR_DIR / "*.png")))[:NUM_CALIBRATION]

    def build_inputs(image, prompt="Describe the image concisely:"):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        return processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")

    # Phase 1: Per-layer SVD extraction
    layer_indices = [num_layers + off for off in SCAN_LAYER_OFFSETS]
    layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
    print(f"Scan layers: {layer_indices}")

    entity_h = {li: [] for li in layer_indices}
    captures = {}
    handles = []
    for li in layer_indices:
        cap = LayerCapture()
        handles.append(decoder_layers[li].register_forward_hook(cap))
        captures[li] = cap

    t0 = time.time()
    for idx, img_path in enumerate(cal_paths):
        image = Image.open(img_path).convert("RGB")
        image = image.filter(ImageFilter.GaussianBlur(radius=20))
        inputs = build_inputs(image)
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        for cap in captures.values():
            cap.reset()
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_CAL, do_sample=False)
        gen_ids = out[0, input_len:]
        ent_pos = locate_entity_position(gen_ids, tokenizer)
        cap_step = max(0, ent_pos - 1)

        hidden_dim = None
        for li in layer_indices:
            cap = captures[li]
            if cap_step < len(cap.step_states):
                entity_h[li].append(cap.step_states[cap_step])
                if hidden_dim is None:
                    hidden_dim = cap.step_states[cap_step].shape[-1]
            else:
                entity_h[li].append(torch.zeros(1, hidden_dim or 2048))

        del inputs, out, gen_ids, image
        flush_vram()
        if (idx + 1) % 10 == 0:
            print(f"  Cal [{idx+1}/{len(cal_paths)}] [{time.time()-t0:.0f}s]")

    for h in handles:
        h.remove()

    layer_svd = {}
    for li in layer_indices:
        H = torch.cat(entity_h[li], dim=0)
        H_mean = H.mean(0, keepdim=True)
        R = H - H_mean
        U, S, Vt = torch.linalg.svd(R, full_matrices=False)
        V_bias = Vt[:K, :]
        total_var = (S ** 2).sum()
        evr = ((S[:K] ** 2).sum() / total_var).item()
        L_prior = torch.sqrt((S[:K] ** 2).sum()).item() / H.shape[0] ** 0.5
        layer_svd[li] = {"V_bias": V_bias, "evr": evr, "L_prior": L_prior}
    del entity_h
    flush_vram()

    # Phase 2: POPE evaluation per layer
    pope_data = []
    with open(POPE_FILE) as f:
        for line in f:
            pope_data.append(json.loads(line.strip()))
    pope_data = pope_data[:POPE_LIMIT]
    pope_images = {}
    for s in pope_data:
        img_name = s["image"]
        if img_name not in pope_images:
            img = load_pope_image(img_name)
            if img:
                pope_images[img_name] = img.filter(ImageFilter.GaussianBlur(radius=15))
    pope_data = [s for s in pope_data if s["image"] in pope_images]
    print(f"POPE samples: {len(pope_data)}")

    results_rows = []
    configs = [("base", None)] + [(f"L{num_layers+off}", num_layers+off) for off in SCAN_LAYER_OFFSETS if 0 <= num_layers+off < num_layers]

    for config_name, layer_idx in configs:
        hook = None
        if layer_idx is not None and layer_idx in layer_svd:
            svd = layer_svd[layer_idx]
            hook = LayerAOSPHook(decoder_layers, layer_idx, svd["V_bias"], svd["L_prior"], alpha=0.5, mu=0, burn_in=0)

        tp = fp = tn = fn = 0
        total_nll = 0.0
        total_tokens = 0
        total_intv = 0

        for si, sample in enumerate(pope_data):
            img_name = sample["image"]
            question = sample["question"]
            gt = sample["ground_truth"].strip().lower()
            image = pope_images[img_name]
            prompt_text = f"{question} Answer with yes or no."
            inputs = build_inputs(image, prompt_text)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            else:
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            if hook:
                hook.reset()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_POPE, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
            gen_ids = out.sequences[0, input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

            if out.scores:
                for step_idx, logits in enumerate(out.scores):
                    if step_idx < len(gen_ids):
                        log_probs = F.log_softmax(logits[0], dim=-1)
                        total_nll -= log_probs[gen_ids[step_idx]].item()
                        total_tokens += 1

            pred = "yes" if "yes" in gen_text else "no"
            if gt == "yes" and pred == "yes": tp += 1
            elif gt == "no" and pred == "yes": fp += 1
            elif gt == "no" and pred == "no": tn += 1
            elif gt == "yes" and pred == "no": fn += 1

            if hook:
                total_intv += hook.interventions
            del inputs, out, gen_ids
            if (si + 1) % 10 == 0:
                flush_vram()

        count = len(pope_data)
        acc = (tp + tn) / max(count, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        ppl = torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item()
        evr = layer_svd[layer_idx]["evr"] if (layer_idx is not None and layer_idx in layer_svd) else 0.0

        results_rows.append({
            "config": config_name,
            "layer_idx": layer_idx if layer_idx is not None else -1,
            "evr": round(evr, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "ppl": round(ppl, 2),
            "total_interventions": total_intv,
        })
        print(f"  {config_name}: F1={f1:.4f} PPL={ppl:.2f} Intv={total_intv}")

        if hook:
            hook.remove()
        flush_vram()

    del model
    flush_vram()

    csv_path = OUT_LOGS / f"layer_sensitivity_qwen3vl_{model_size}.csv"
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        w.writeheader()
        w.writerows(results_rows)
    print(f"\nResults → {csv_path}")
    return csv_path, results_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["2b", "8b", "both"], default="2b")
    args = parser.parse_args()

    models_to_run = []
    if args.model in ("2b", "both"):
        models_to_run.append(("Qwen/Qwen3-VL-2B-Instruct", "2b"))
    if args.model in ("8b", "both"):
        models_to_run.append(("Qwen/Qwen3-VL-8B-Instruct", "8b"))

    for model_id, size in models_to_run:
        run_model(model_id, size)

    print("\nDone.")


if __name__ == "__main__":
    main()
