"""
Fatal Ablation: A-OSP WITHOUT Scale Preservation
==================================================
Removes the energy compensation (H_proj / ||H_proj|| * ||H||) from
the orthogonal projection, leaving only raw subtraction:
    H' = H - alpha * sum <H, v_i> v_i          # NO rescaling!

This causes L2 norm decay → activation collapse → PPL explosion → word salad.
Captures per-step residual stream L2 norms and generates a crash log with
the gibberish output as irrefutable evidence for the necessity of scale
preservation (Paper §4.4 "Energy Scale Preservation").

Usage:
  python run_ablation_no_scale.py
"""

import os, sys, json, io, csv, time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
MODEL_PATH = PROJECT / "models" / "Qwen2-VL-7B-Instruct"
V_MATRIX_PATH = PROJECT / "models" / "V_matrix.pt"
MANIFEST_PATH = PROJECT / "data" / "blurred_calibration" / "calibration_manifest.json"
OUTPUT_DIR = PROJECT / "data" / "ablation"
COCO_BASE_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

INTERVENTION_LAYER = 24
MAX_NEW_TOKENS = 128
PROMPT = "Describe the image in detail:"
NUM_ABLATION_IMAGES = 5
BLUR_DIR = PROJECT / "data" / "blurred_calibration" / "blur"


# ═══════════════════════════════════════════════════════════════
#  BROKEN Hook: Orthogonal Subtraction WITHOUT Scale Preservation
# ═══════════════════════════════════════════════════════════════
class BrokenAOSPHook:
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  FATAL DEFECT: scale compensation line is REMOVED.      ║
    ║  H' = H - alpha * proj @ V   (raw subtraction only)    ║
    ║  Expected: L2 norm decay → activation collapse → PPL↑  ║
    ╚══════════════════════════════════════════════════════════╝
    Records per-step: L2 norm before/after intervention, to show the
    catastrophic norm decay across generation steps.
    """

    def __init__(self, model, decoder_layers, V_bias, L_prior,
                 alpha=0.5, mu=1.5, beta=0.9, eps=0.1):
        self.model = model
        self.V = V_bias.clone()
        self.alpha, self.mu, self.beta, self.eps = alpha, mu, beta, eps
        self.L_bar = L_prior
        self.L_prior_val = L_prior
        self.t = 0
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0

        self.norm_log: list[dict] = []

        self._handle = decoder_layers[INTERVENTION_LAYER].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        h = out[0]
        if not self._prefill_done:
            self._prefill_done = True
            return
        if h.shape[1] != 1:
            return

        self.t += 1
        H = h[:, -1, :]          # [1, D]
        V_dev = self.V.to(H.device, dtype=H.dtype)

        norm_before = H.norm(dim=-1).item()

        # ── Simplified burn-in (skip entropy, use fixed N=5 for ablation) ──
        if self.t <= 5:
            self.norm_log.append({
                "step": self.t, "norm_before": norm_before,
                "norm_after": norm_before, "triggered": False, "burn_in": True,
            })
            return

        proj = H @ V_dev.T
        L_t = torch.sqrt((proj ** 2).sum(-1)).item()
        triggered = L_t > self.mu * self.L_bar

        if triggered:
            # ╔════════════════════════════════════════════╗
            # ║  THE FATAL LINE: no scale compensation!    ║
            # ║  Compare with correct version:             ║
            # ║    H' = H_proj / ||H_proj|| * ||H||        ║
            # ║  Here we just do:                          ║
            # ║    H' = H_proj  (raw, norm-decaying)       ║
            # ╚════════════════════════════════════════════╝
            H_proj = H - self.alpha * (proj @ V_dev)
            h[:, -1, :] = H_proj          # <-- NO RESCALING
            self.interventions += 1
            norm_after = H_proj.norm(dim=-1).item()
        else:
            norm_after = norm_before
            if self.t > 5:
                self.L_bar = self.beta * self.L_bar + (1 - self.beta) * L_t

        self.norm_log.append({
            "step": self.t, "norm_before": norm_before,
            "norm_after": norm_after, "triggered": triggered, "burn_in": False,
            "L_t": L_t, "L_bar": self.L_bar,
            "norm_decay_pct": (1.0 - norm_after / max(norm_before, 1e-8)) * 100,
        })

    def reset(self):
        self.norm_log.clear()
        self.t = 0
        self.L_bar = self.L_prior_val
        self.N_adaptive = None
        self.prev_entropy = None
        self._prefill_done = False
        self.interventions = 0

    def remove(self):
        self._handle.remove()


# ═══════════════════════════════════════════════════════════════
#  Also run the CORRECT A-OSP for side-by-side comparison
# ═══════════════════════════════════════════════════════════════
class CorrectAOSPHook:
    """Correct version with scale preservation for comparison."""

    def __init__(self, model, decoder_layers, V_bias, L_prior,
                 alpha=0.5, mu=1.5, beta=0.9):
        self.model = model
        self.V = V_bias.clone()
        self.alpha, self.mu, self.beta = alpha, mu, beta
        self.L_bar = L_prior
        self.L_prior_val = L_prior
        self.t = 0
        self._prefill_done = False
        self.interventions = 0
        self.norm_log: list[dict] = []
        self._handle = decoder_layers[INTERVENTION_LAYER].register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        h = out[0]
        if not self._prefill_done:
            self._prefill_done = True; return
        if h.shape[1] != 1: return
        self.t += 1
        H = h[:, -1, :]
        V_dev = self.V.to(H.device, dtype=H.dtype)
        norm_before = H.norm(-1).item()
        if self.t <= 5:
            self.norm_log.append({"step": self.t, "norm_before": norm_before,
                                  "norm_after": norm_before, "triggered": False})
            return
        proj = H @ V_dev.T
        L_t = torch.sqrt((proj ** 2).sum(-1)).item()
        if L_t > self.mu * self.L_bar:
            H_proj = H - self.alpha * (proj @ V_dev)
            on = torch.clamp(H.norm(-1, keepdim=True), min=1e-6)
            pn = torch.clamp(H_proj.norm(-1, keepdim=True), min=1e-6)
            h[:, -1, :] = H_proj / pn * on    # <-- CORRECT: scale preserved
            self.interventions += 1
            self.norm_log.append({"step": self.t, "norm_before": norm_before,
                                  "norm_after": on.item(), "triggered": True})
        else:
            self.L_bar = self.beta * self.L_bar + (1 - self.beta) * L_t
            self.norm_log.append({"step": self.t, "norm_before": norm_before,
                                  "norm_after": norm_before, "triggered": False})

    def reset(self):
        self.norm_log.clear(); self.t = 0; self.L_bar = self.L_prior_val
        self._prefill_done = False; self.interventions = 0

    def remove(self):
        self._handle.remove()


# ═══════════════════════════════════════════════════════════════
def download_coco(cid):
    import requests
    url = COCO_BASE_URL.format(cid)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def build_inputs(image, processor):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(msgs)
        return processor(text=[text], images=imgs, videos=vids,
                         return_tensors="pt", padding=True)
    except ImportError:
        return processor(text=[text], images=[image],
                         return_tensors="pt", padding=True)


@torch.no_grad()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "ablation_no_scale_crash.log"

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print("Loading model ...")
    try:
        import flash_attn; attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation=attn)
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model.eval()
    decoder_layers = model.model.language_model.layers
    tokenizer = processor.tokenizer

    ckpt = torch.load(str(V_MATRIX_PATH), map_location="cpu", weights_only=True)
    V_bias = ckpt["V_bias"]
    L_prior = ckpt.get("L_prior", 100.0)

    # Use BLURRED images to maximize intervention frequency (language prior dominates)
    blur_files = sorted(BLUR_DIR.glob("*.jpg"))[:NUM_ABLATION_IMAGES]
    manifest = json.load(open(MANIFEST_PATH))
    coco_ids = [manifest["items"][i]["coco_image_id"] for i in range(NUM_ABLATION_IMAGES)]

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("FATAL ABLATION: A-OSP WITHOUT SCALE PRESERVATION")
    log_lines.append("=" * 80)
    log_lines.append("Condition: BLURRED images (force high intervention rate)")
    log_lines.append("The broken version removes: H' = H_proj / ||H_proj|| * ||H||")
    log_lines.append("Expected: compounding L2 norm decay → activation collapse → word salad")
    log_lines.append("=" * 80)

    norm_decay_csv_path = OUTPUT_DIR / "ablation_norm_decay.csv"
    csv_rows = []

    for idx, bf in enumerate(blur_files):
        image = Image.open(bf).convert("RGB")
        cid = coco_ids[idx]
        inputs = build_inputs(image, processor).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        log_lines.append(f"\n{'─'*80}")
        log_lines.append(f"Image {idx+1}/{NUM_ABLATION_IMAGES}: Blurred COCO ID {cid} ({bf.name})")
        log_lines.append(f"{'─'*80}")

        # ── 1. Base generation (no intervention) ──
        out_base = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        base_text = tokenizer.decode(out_base[0, input_len:], skip_special_tokens=True)
        log_lines.append(f"\n[BASE] {base_text}")

        # ── 2. Correct A-OSP (aggressive: mu=1.1 to force frequent triggers) ──
        correct_hook = CorrectAOSPHook(model, decoder_layers, V_bias, L_prior, mu=1.1)
        out_correct = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        correct_text = tokenizer.decode(out_correct[0, input_len:], skip_special_tokens=True)
        log_lines.append(f"\n[A-OSP CORRECT, mu=1.1] (interventions={correct_hook.interventions}) {correct_text}")

        correct_norms = correct_hook.norm_log
        correct_hook.remove()

        # ── 3. BROKEN A-OSP (no scale, aggressive: mu=1.1) ──
        broken_hook = BrokenAOSPHook(model, decoder_layers, V_bias, L_prior, mu=1.1)
        out_broken = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        broken_text = tokenizer.decode(out_broken[0, input_len:], skip_special_tokens=True)
        log_lines.append(f"\n[A-OSP NO-SCALE] (interventions={broken_hook.interventions}) {broken_text}")

        broken_norms = broken_hook.norm_log
        broken_hook.remove()

        # ── Norm decay analysis ──
        if broken_norms:
            first_norm = broken_norms[0]["norm_before"]
            last_norm = broken_norms[-1]["norm_after"]
            decay_pct = (1.0 - last_norm / max(first_norm, 1e-8)) * 100
            triggered_norms = [n for n in broken_norms if n.get("triggered")]
            if triggered_norms:
                avg_per_step_decay = sum(n.get("norm_decay_pct", 0)
                                         for n in triggered_norms) / len(triggered_norms)
            else:
                avg_per_step_decay = 0

            log_lines.append(f"\n  [NORM ANALYSIS]")
            log_lines.append(f"    First norm:        {first_norm:.2f}")
            log_lines.append(f"    Last norm:         {last_norm:.2f}")
            log_lines.append(f"    Total decay:       {decay_pct:.1f}%")
            log_lines.append(f"    Avg decay/trigger: {avg_per_step_decay:.2f}%")
            log_lines.append(f"    Interventions:     {broken_hook.interventions}")

            for n in broken_norms:
                csv_rows.append({
                    "coco_id": cid, "step": n["step"],
                    "norm_before": n["norm_before"], "norm_after": n["norm_after"],
                    "triggered": n.get("triggered", False),
                    "mode": "no_scale",
                })
            for n in correct_norms:
                csv_rows.append({
                    "coco_id": cid, "step": n["step"],
                    "norm_before": n["norm_before"], "norm_after": n["norm_after"],
                    "triggered": n.get("triggered", False),
                    "mode": "correct",
                })

    # ── Save log ──
    log_text = "\n".join(log_lines)
    with open(log_path, "w") as f:
        f.write(log_text)
    print(log_text)

    # ── Save norm decay CSV ──
    with open(norm_decay_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["coco_id", "step", "norm_before",
                                           "norm_after", "triggered", "mode"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\nCrash log → {log_path}")
    print(f"Norm decay CSV → {norm_decay_csv_path}")


if __name__ == "__main__":
    main()
