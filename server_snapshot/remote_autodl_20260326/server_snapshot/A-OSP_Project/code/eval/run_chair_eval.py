#!/usr/bin/env python3
"""
CHAIR Captioning Evaluation Pipeline for A-OSP
================================================
Generates open-ended image captions via Qwen3-VL-8B then computes
CHAIR_s / CHAIR_i / Recall using the standalone CHAIR metric.

Dependencies: COCO val2014 instance annotations (instances_val2014.json).
If not present, downloads automatically.

Usage:
    python run_chair_eval.py --mode base   --limit 50
    python run_chair_eval.py --mode aosp   --limit 50
    python run_chair_eval.py --mode score  --cap_file <jsonl>
"""

import argparse
import gc
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import (
    append_jsonl,
    load_completed_ids,
    load_jsonl,
    load_qwen2vl,
    aggressive_vram_cleanup,
)

PROJECT = Path("/root/autodl-tmp/A-OSP_Project")
CHAIR_DIR = PROJECT / "data" / "chair" / "CHAIR-metric-standalone"
COCO_ANN_DIR = PROJECT / "data" / "chair" / "coco_annotations"
COCO_IMAGES_DIR = PROJECT / "data" / "coco_val2014"
OUT_DIR = PROJECT / "logs" / "eval_results"

DEFAULT_MODEL = str(PROJECT / "models" / "Qwen3-VL-8B-Instruct")
# Use Qwen3-VL-specific V_matrix; fall back to base if not yet extracted
_q3_vmatrix = PROJECT / "models" / "V_matrix_q3.pt"
_q2_vmatrix = PROJECT / "models" / "V_matrix.pt"
DEFAULT_V_MATRIX = str(_q3_vmatrix if _q3_vmatrix.exists() else _q2_vmatrix)

CAPTION_PROMPT = "Please describe everything you can see in this image in detail."

# ---------------------------------------------------------------------------
# Minimal CHAIR implementation (no pattern.en dependency)
# ---------------------------------------------------------------------------

def _simple_singularize(word: str) -> str:
    """Rule-based English singularizer for COCO object classes."""
    exceptions = {
        "people": "person", "men": "man", "women": "woman", "children": "child",
        "mice": "mouse", "teeth": "tooth", "feet": "foot", "oxen": "ox",
        "knives": "knife", "lives": "life", "leaves": "leaf", "halves": "half",
    }
    if word in exceptions:
        return exceptions[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 3:
        return word[:-3] + "f"
    if word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        return word[:-2]
    if word.endswith("ches") or word.endswith("shes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _load_synonyms():
    synonyms_raw = [
        "person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger",
        "bicycle, bike, unicycle, minibike, trike",
        "car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi",
        "motorcycle, scooter, motor bike, motor cycle, motorbike, moped",
        "airplane, jetliner, plane, air plane, monoplane, aircraft, jet, airbus, biplane, seaplane",
        "bus, minibus, trolley",
        "train, locomotive, tramway, caboose",
        "truck, pickup, lorry, hauler, firetruck",
        "boat, ship, liner, sailboat, motorboat, dinghy, powerboat, canoe, skiff, yacht, kayak, vessel, rowboat",
        "traffic light, street light, traffic signal, stop light, streetlight, stoplight",
        "fire hydrant, hydrant",
        "stop sign", "parking meter",
        "bench, pew",
        "bird, owl, seagull, goose, duck, falcon, robin, sparrow, eagle, crow, pigeon",
        "cat, kitten, feline, tabby",
        "dog, puppy, beagle, pup, chihuahua, canine, pitbull, poodle, labrador, bulldog, husky",
        "horse, colt, pony, stallion, mare, foal, palomino, mustang",
        "sheep, lamb, ram, goat, ewe",
        "cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, bison",
        "elephant", "bear, panda", "zebra", "giraffe",
        "backpack, knapsack",
        "umbrella",
        "handbag, wallet, purse, briefcase",
        "tie, bow, bow tie",
        "suitcase, suit case, luggage",
        "frisbee", "skis, ski", "snowboard",
        "sports ball, ball", "kite",
        "baseball bat", "baseball glove", "skateboard",
        "surfboard, longboard, skimboard, shortboard",
        "tennis racket, racket",
        "bottle", "wine glass", "cup",
        "fork", "knife, pocketknife, knive", "spoon",
        "bowl, container",
        "banana", "apple",
        "sandwich, burger, sub, cheeseburger, hamburger",
        "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut, doughnut, bagel",
        "cake, cheesecake, cupcake, pancake",
        "chair, seat, stool",
        "couch, sofa, recliner, futon, loveseat",
        "potted plant, houseplant",
        "bed",
        "dining table, table, desk",
        "toilet, urinal, commode, lavatory",
        "tv, monitor, television",
        "laptop, computer, notebook, netbook, macbook",
        "mouse", "remote", "keyboard",
        "cell phone, mobile phone, phone, cellphone, smartphone",
        "microwave",
        "oven, stovetop, stove",
        "toaster", "sink",
        "refrigerator, fridge, freezer",
        "book", "clock", "vase", "scissors",
        "teddy bear, teddybear",
        "hair drier, hairdryer",
        "toothbrush",
    ]
    mscoco_objects = []
    inverse_synonym_dict = {}
    for line in synonyms_raw:
        synonyms = [s.strip() for s in line.split(",")]
        canonical = synonyms[0]
        mscoco_objects.extend(synonyms)
        for s in synonyms:
            inverse_synonym_dict[s] = canonical
    return set(mscoco_objects), inverse_synonym_dict


_MSCOCO_OBJECTS, _INVERSE_SYNONYM = _load_synonyms()

_DOUBLE_WORDS = {
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
    "hair drier", "potted plant", "bow tie", "laptop computer",
}


def caption_to_node_words(caption: str) -> list:
    tokens = re.findall(r"[a-z]+", caption.lower())
    tokens = [_simple_singularize(t) for t in tokens]
    merged, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = tokens[i] + " " + tokens[i + 1]
            if pair in _DOUBLE_WORDS:
                merged.append(pair)
                i += 2
                continue
        merged.append(tokens[i])
        i += 1
    node_words = []
    for w in merged:
        if w in _MSCOCO_OBJECTS:
            node_words.append(_INVERSE_SYNONYM[w])
    return node_words


def build_imid_to_objects(instances_json_path: str) -> dict:
    print(f"[CHAIR] Loading instances from {instances_json_path} ...")
    data = json.load(open(instances_json_path))
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    imid_to_objects: dict = {}
    for ann in data["annotations"]:
        imid = ann["image_id"]
        name = id_to_name.get(ann["category_id"], "")
        node = _INVERSE_SYNONYM.get(name, "")
        if node:
            imid_to_objects.setdefault(imid, set()).add(node)
    print(f"[CHAIR] Loaded {len(imid_to_objects)} image records.")
    return imid_to_objects


def compute_chair_metrics(captions: list[dict], imid_to_objects: dict) -> dict:
    """captions: list of {image_id: int, caption: str}"""
    n_caps = 0
    n_hallucinated = 0
    n_hal_words = 0
    n_coco_words = 0
    n_recall_gt = 0
    n_gt_total = 0
    sentences = []

    for item in captions:
        imid = int(item["image_id"])
        caption = item["caption"]
        node_words = caption_to_node_words(caption)
        gt_objects = imid_to_objects.get(imid, set())

        hallucinated_words = [w for w in node_words if w not in gt_objects]
        recall_words = set(w for w in node_words if w in gt_objects)

        hallucinated = len(hallucinated_words) > 0
        n_caps += 1
        if hallucinated:
            n_hallucinated += 1
        n_hal_words += len(hallucinated_words)
        n_coco_words += len(node_words)
        n_recall_gt += len(recall_words)
        n_gt_total += len(gt_objects)

        sentences.append({
            "image_id": imid,
            "caption": caption,
            "hallucinated_words": hallucinated_words,
            "gt_objects": list(gt_objects),
            "generated_node_words": node_words,
            "CHAIRs": int(hallucinated),
            "CHAIRi": len(hallucinated_words) / max(len(node_words), 1),
            "Recall": len(recall_words) / max(len(gt_objects), 1),
        })

    chair_s = n_hallucinated / max(n_caps, 1)
    chair_i = n_hal_words / max(n_coco_words, 1)
    recall = n_recall_gt / max(n_gt_total, 1)

    return {
        "overall": {
            "CHAIRs": chair_s,
            "CHAIRi": chair_i,
            "Recall": recall,
            "n_caps": n_caps,
            "n_hallucinated": n_hallucinated,
        },
        "sentences": sentences,
    }


# ---------------------------------------------------------------------------
# COCO annotations download helper
# ---------------------------------------------------------------------------

def ensure_coco_instances(ann_dir: Path) -> str:
    instance_path = ann_dir / "instances_val2014.json"
    if instance_path.exists():
        return str(instance_path)
    ann_dir.mkdir(parents=True, exist_ok=True)
    print("[CHAIR] Downloading COCO val2014 instance annotations (~25 MB)...")
    import urllib.request
    url = "http://images.cocodataset.org/annotations/instances_val2014.json"
    urllib.request.urlretrieve(url, str(instance_path))
    print(f"[CHAIR] Saved → {instance_path}")
    return str(instance_path)


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def find_image(image_id: int) -> str | None:
    fname_base = f"COCO_val2014_{image_id:012d}"
    for ext in [".jpg", ".png", ".jpeg"]:
        p = COCO_IMAGES_DIR / (fname_base + ext)
        if p.exists():
            return str(p)
    for ext in [".jpg", ".jpeg", ".png"]:
        for cid_str in [str(image_id), fname_base]:
            p = COCO_IMAGES_DIR / (cid_str + ext)
            if p.exists():
                return str(p)
    return None


def generate_captions(model, processor, image_ids: list[int],
                      output_path: Path, aosp_handle=None,
                      limit: int = 0) -> None:
    from PIL import Image, ImageFile
    import torch
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    completed = load_completed_ids(str(output_path))
    if completed:
        print(f"[Resume] {len(completed)} already done.")
    if limit:
        image_ids = image_ids[:limit]
    pending = [i for i in image_ids if i not in completed]
    print(f"[Captioning] {len(pending)} pending / {len(image_ids)} total")

    for idx, image_id in enumerate(pending):
        img_path = find_image(image_id)
        if img_path is None:
            print(f"  [SKIP] image_id={image_id} not found")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            max_px = 1024 * 1024
            w, h = img.size
            if w * h > max_px:
                scale = (max_px / (w * h)) ** 0.5
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            if aosp_handle:
                aosp_handle.reset()

            t0 = time.time()
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            latency = time.time() - t0
            n_new = ids.shape[1] - inputs["input_ids"].shape[1]
            caption = processor.decode(ids[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip()
            ic = aosp_handle.intervention_count if aosp_handle else 0

            record = {
                "image_id": image_id,
                "caption": caption,
                "generation_length": n_new,
                "latency_s": round(latency, 3),
                "intervention_count": ic,
                "question_id": image_id,
            }
            append_jsonl(str(output_path), record)

        except Exception as e:
            print(f"  [ERROR] image_id={image_id}: {e}")
        finally:
            del inputs
            gc.collect()
            torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(pending)}] image_id={image_id} | "
                  f"len={record.get('generation_length',0)} | ic={ic}")

    print(f"[Captioning] Done → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "aosp", "score"], default="base")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--v_matrix", default=DEFAULT_V_MATRIX)
    p.add_argument("--limit", type=int, default=50,
                   help="Number of COCO val images to caption (0=all)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--cap_file", type=str, default="",
                   help="For --mode score: JSONL of captions to score")
    p.add_argument("--output_dir", default=str(OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mode_tag = args.mode
    if args.limit:
        mode_tag += f"_n{args.limit}"

    if args.mode == "score":
        cap_file = args.cap_file
        if not cap_file:
            print("Error: --cap_file required for --mode score")
            sys.exit(1)
    else:
        cap_file = str(OUT_DIR / f"chair_{mode_tag}_captions.jsonl")

    ann_path = ensure_coco_instances(COCO_ANN_DIR)
    imid_to_objects = build_imid_to_objects(ann_path)

    if args.mode != "score":
        print(f"\n[Loader] Model: {args.model_path}")
        model, processor = load_qwen2vl(args.model_path)

        aosp_handle = None
        if args.mode == "aosp":
            import sys as _sys
            _sys.path.insert(0, str(PROJECT / "code"))
            from aosp_hook import apply_aosp_hook
            aosp_handle = apply_aosp_hook(
                model, args.v_matrix,
                alpha=args.alpha, mu=args.mu, beta=args.beta,
            )

        image_ids = sorted(imid_to_objects.keys())
        generate_captions(
            model, processor, image_ids,
            Path(cap_file), aosp_handle, limit=args.limit,
        )

        if aosp_handle:
            aosp_handle.remove()
        del model, processor
        aggressive_vram_cleanup()

    captions = load_jsonl(cap_file)
    print(f"\n[CHAIR] Scoring {len(captions)} captions...")
    results = compute_chair_metrics(captions, imid_to_objects)
    ov = results["overall"]

    score_path = cap_file.replace(".jsonl", "_chair_scores.json")
    with open(score_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"CHAIR Evaluation Results ({mode_tag})")
    print("=" * 50)
    print(f"  CHAIRs: {ov['CHAIRs']:.4f}  ({ov['n_hallucinated']}/{ov['n_caps']} hallucinatory caps)")
    print(f"  CHAIRi: {ov['CHAIRi']:.4f}")
    print(f"  Recall: {ov['Recall']:.4f}")
    print(f"  Saved → {score_path}")


if __name__ == "__main__":
    main()
