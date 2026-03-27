"""
A-OSP Subspace Extractor — Distribution-Shift Guided Bias Subspace Extraction
==============================================================================
Hook into layer -4 of Qwen2-VL-7B. Under weak visual conditions (Gaussian-blurred
calibration images), let the model free-generate and intercept the hidden state of 
the LAST TEXT TOKEN at the precise moment an entity noun is predicted.
Mean-center → SVD → save top-K right singular vectors as bias subspace basis.

Theory anchor: blurred images strip fine-grained visual grounding; the model's
generation trajectory collapses onto its structural language inertia manifold.
The top-K singular vectors of mean-centered hidden states capture the dominant
directions of this unconditional autoregressive degradation.

RED LINE: We extract TEXT-SIDE hidden states only. NEVER pool visual tokens.
"""

import os
import glob
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

# ========================== Configuration ==========================
MODEL_PATH = "/root/autodl-tmp/A-OSP_Project/models/Qwen2-VL-7B-Instruct"
BLUR_DIR = "/root/autodl-tmp/A-OSP_Project/data/blurred_calibration"
OUTPUT_PATH = "/root/autodl-tmp/A-OSP_Project/models/V_matrix.pt"
TARGET_LAYER_OFFSET = -4          # 4th layer from the end (semantic decoupling zone)
K = 20                            # number of principal components
MAX_NEW_TOKENS = 64               # free-generation budget per image
PROMPT = "Describe the image concisely:"

# ========================== Entity Detection ==========================
DETERMINERS = frozenset({"a", "an", "the", "some", "one", "two", "three",
                         "several", "many", "this", "that", "these", "those"})

TEMPLATE_WORDS = frozenset({
    "image", "picture", "photo", "photograph", "scene", "view", "shot",
    "blurred", "blurry", "unclear", "indistinct", "hazy",
    "it", "is", "shows", "depicts", "appears", "seems", "features",
    "there", "are", "was", "were", "has", "have", "been",
    "not", "very", "quite", "rather", "somewhat",
})


def locate_entity_position(generated_ids, tokenizer):
    """
    Heuristic: find the first concrete entity noun in the generated sequence.
    Strategy — scan for the first token following a determiner (a/an/the/...)
    that is NOT a template/meta word (image, picture, blurry, etc.).
    Falls back to the first content-rich token in the generation.
    Returns 0-indexed position in the generated token list.
    """
    decoded_tokens = [
        tokenizer.decode([tid], skip_special_tokens=True).strip().lower()
        for tid in generated_ids
    ]

    # Pass 1: determiner + concrete noun pattern (skip template words)
    for i, tok in enumerate(decoded_tokens):
        if tok in DETERMINERS and (i + 1) < len(decoded_tokens):
            candidate = decoded_tokens[i + 1]
            if (len(candidate) > 2 and candidate.isalpha()
                    and candidate not in TEMPLATE_WORDS):
                return i + 1

    # Pass 2: first token that looks like a concrete noun (>3 chars, alphabetic,
    # not a template word), typically where the model "makes stuff up"
    for i, tok in enumerate(decoded_tokens):
        if (len(tok) > 3 and tok.isalpha()
                and tok not in TEMPLATE_WORDS and tok not in DETERMINERS):
            return i

    return max(1, len(decoded_tokens) // 3)


# ========================== Hook Machinery ==========================
class LayerHiddenStateCapture:
    """
    Captures the last text token's hidden state at each autoregressive
    generation step from a specific transformer layer.
    
    During prefill the full input sequence passes through — we skip it.
    During each generation step only 1 new token is processed (KV-cached);
    its hidden state at the hooked layer is exactly "the last text token's
    representation that drives the next-token prediction".
    """

    def __init__(self):
        self.step_states: list[torch.Tensor] = []
        self._prefill_done = False

    def reset(self):
        self.step_states.clear()
        self._prefill_done = False

    def __call__(self, module, inp, out):
        hidden = out[0]                           # [batch, seq_len_or_1, D]
        if not self._prefill_done:
            self._prefill_done = True
            return
        h = hidden[:, -1, :].detach().float()     # [1, D]
        self.step_states.append(h.cpu())


# ========================== Main Pipeline ==========================
@torch.no_grad()
def extract_bias_subspace():
    # ---- Load model & processor ----
    print(f"Loading model from {MODEL_PATH} ...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("[WARN] flash_attn not found, falling back to SDPA (PyTorch native)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    tokenizer = processor.tokenizer

    # ---- Determine target layer ----
    # Qwen2-VL nests decoder layers under model.language_model.layers
    decoder_layers = model.model.language_model.layers
    num_layers = len(decoder_layers)
    target_idx = num_layers + TARGET_LAYER_OFFSET
    print(f"Total layers: {num_layers}, hooking layer index {target_idx} (offset {TARGET_LAYER_OFFSET})")

    # ---- Register hook ----
    capture = LayerHiddenStateCapture()
    handle = decoder_layers[target_idx].register_forward_hook(capture)

    # ---- Collect blurred images (Agent 3 places them in blur/ subdirectory) ----
    image_paths = sorted(
        glob.glob(os.path.join(BLUR_DIR, "blur", "*.jpg"))
        + glob.glob(os.path.join(BLUR_DIR, "blur", "*.png"))
        + glob.glob(os.path.join(BLUR_DIR, "blur", "*.jpeg"))
    )
    if not image_paths:
        image_paths = sorted(
            glob.glob(os.path.join(BLUR_DIR, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(BLUR_DIR, "**", "*.png"), recursive=True)
        )
    image_paths = image_paths[:200]
    assert len(image_paths) > 0, f"No images found in {BLUR_DIR}"
    print(f"Found {len(image_paths)} calibration images.")

    all_hidden: list[torch.Tensor] = []   # each element: [1, D]

    for idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")

        # ---- Build Qwen2-VL chat message ----
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if HAS_QWEN_VL_UTILS:
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=img_inputs, videos=vid_inputs,
                return_tensors="pt", padding=True
            )
        else:
            inputs = processor(
                text=[text], images=[image],
                return_tensors="pt", padding=True
            )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # ---- Reset hook & free-generate ----
        capture.reset()
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        generated_ids = output_ids[0, input_len:]

        # ---- Locate entity prediction moment ----
        #   locate_entity_position returns the position of the entity TOKEN
        #   in the generated sequence (0-indexed).
        #   The hidden state that DRIVES this prediction lives one step earlier:
        #     step_states[entity_pos - 1]  predicts  generated_token[entity_pos]
        entity_pos = locate_entity_position(generated_ids, tokenizer)
        capture_idx = max(0, entity_pos - 1)

        if capture_idx < len(capture.step_states):
            all_hidden.append(capture.step_states[capture_idx])        # [1, D]
        elif len(capture.step_states) > 0:
            all_hidden.append(capture.step_states[len(capture.step_states) // 2])

        if (idx + 1) % 20 == 0 or idx == 0:
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"  [{idx+1:>3}/{len(image_paths)}] entity_pos={entity_pos} | "
                  f"{gen_text[:80]}...")

    handle.remove()
    print(f"\nCollected {len(all_hidden)} hidden states for SVD.")

    # ====================== Mean-Center + SVD ======================
    H = torch.cat(all_hidden, dim=0)               # [N, D]
    H_mean = H.mean(dim=0, keepdim=True)            # [1, D]
    R = H - H_mean                                  # [N, D]  mean-centered residuals

    U, S, Vt = torch.linalg.svd(R, full_matrices=False)  # Vt: [N, D]

    V_bias = Vt[:K, :]                              # [K, D]  top-K right singular vectors

    total_var = (S ** 2).sum()
    explained = (S[:K] ** 2).sum() / total_var
    print(f"Top-{K} Explained Variance Ratio (EVR): {explained:.4f}")
    print(f"  (>0.85 validates local-linearity hypothesis at deep layers)")

    # ====================== Compute L_prior (mean projection energy) =====
    L_all = torch.sqrt((H @ V_bias.T).pow(2).sum(dim=-1))  # [N]
    L_prior = L_all.mean().item()
    print(f"L_prior (mean projection energy on calibration set): {L_prior:.2f}")

    # ====================== Persist ======================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    payload = {
        "V_bias": V_bias,                            # [K, D] orthonormal basis of S_bias
        "H_mean": H_mean,                            # [1, D] centering vector
        "singular_values": S[:K],                     # [K]    for diagnostics
        "evr": explained.item(),
        "K": K,
        "layer_idx": target_idx,
        "num_samples": len(all_hidden),
        "L_prior": L_prior,                          # EMA burn-in baseline
    }
    torch.save(payload, OUTPUT_PATH)
    print(f"Saved V_matrix → {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    extract_bias_subspace()
