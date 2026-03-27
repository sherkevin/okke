"""
BRA Evaluation Matrix -- Baseline vs BRA on Multiple Datasets
================================================================
Runs N samples per dataset for each model, comparing vanilla generation
against BRA-augmented generation.  Reports per-dataset metrics aligned
with the paper (POPE F1, CHAIR_s/i, MMBench Acc, MME Perception, etc.).

Usage:
    python3 bra_eval_matrix.py [--n_samples 50] [--model qwen] [--dataset pope]
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import string
import sys
import time
import traceback
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from category_leakage_audit import (
    build_freak_entity_index,
    build_leakage_summary,
    load_catalog_entities,
)

PROJECT = Path("/root/autodl-tmp/BRA_Project")
MODELS_DIR = PROJECT / "models"
DATASETS_DIR = PROJECT / "datasets"
TEMP_CACHE_DIR = PROJECT / ".bra_cache"

sys.path.insert(0, str(PROJECT))

# ── Model registry ──────────────────────────────────────────────────

MODEL_REGISTRY = {
    "qwen3vl2b": {
        "path": MODELS_DIR / "Qwen3-VL-2B-Instruct",
        "hf_class": "Qwen3VLForConditionalGeneration",
        "hf_processor": "AutoProcessor",
        "builder": "qwen3vl",
        "supports_video": True,
    },
    "qwen3vl8b": {
        "path": MODELS_DIR / "Qwen3-VL-8B-Instruct",
        "hf_class": "Qwen3VLForConditionalGeneration",
        "hf_processor": "AutoProcessor",
        "builder": "qwen3vl",
        "supports_video": True,
    },
    "llava7b": {
        "path": MODELS_DIR / "llava-1.5-7b-hf",
        "hf_class": "LlavaForConditionalGeneration",
        "hf_processor": "AutoProcessor",
        "builder": "llava",
        "supports_video": False,
    },
}

# ── Dataset registry ────────────────────────────────────────────────

DATASET_REGISTRY = {
    "pope_adversarial": {
        "loader": "pope",
        "path": DATASETS_DIR / "POPE" / "output" / "coco" / "coco_pope_adversarial.json",
        "image_dir": DATASETS_DIR / "coco2014" / "val2014",
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    "pope_popular": {
        "loader": "pope",
        "path": DATASETS_DIR / "POPE" / "output" / "coco" / "coco_pope_popular.json",
        "image_dir": DATASETS_DIR / "coco2014" / "val2014",
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    "pope_random": {
        "loader": "pope",
        "path": DATASETS_DIR / "POPE" / "output" / "coco" / "coco_pope_random.json",
        "image_dir": DATASETS_DIR / "coco2014" / "val2014",
        "metrics": ["accuracy", "precision", "recall", "f1"],
    },
    "chair": {
        "loader": "chair",
        "image_dir": DATASETS_DIR / "coco2014" / "val2014",
        "ann_path": DATASETS_DIR / "coco2014" / "annotations" / "instances_val2014.json",
        "metrics": ["chair_s", "chair_i", "agl"],
    },
    "mmbench": {
        "loader": "mmbench",
        "path": DATASETS_DIR / "MMBench_EN_hf" / "data" / "dev-00000-of-00001-75b6649fb044d38b.parquet",
        "metrics": ["accuracy"],
    },
    "mme": {
        "loader": "mme",
        "path": DATASETS_DIR / "MME_hf" / "data",
        "metrics": ["perception_score"],
    },
    "hallusionbench": {
        "loader": "hallusionbench",
        "path": DATASETS_DIR / "HallusionBench_hf" / "data" / "image-00000-of-00001.parquet",
        "metrics": ["accuracy"],
    },
    "mmmu": {
        "loader": "mmmu",
        "path": DATASETS_DIR / "MMMU_hf",
        "metrics": ["accuracy"],
    },
    "freak": {
        "loader": "freak",
        "path": DATASETS_DIR / "FREAK_hf" / "data",
        "metrics": ["accuracy"],
    },
    "docvqa": {
        "loader": "docvqa",
        "path_candidates": [
            DATASETS_DIR / "DocVQA_hf" / "data",
            DATASETS_DIR / "docvqa_hf" / "data",
            DATASETS_DIR / "DocVQA",
            DATASETS_DIR / "docvqa",
        ],
        "metrics": ["accuracy"],
    },
    "vidhalluc": {
        "loader": "vidhalluc",
        "path": DATASETS_DIR / "video" / "chaoyuli_VidHalluc",
        "data_dir": DATASETS_DIR / "video" / "chaoyuli_VidHalluc" / "data",
        "metrics": ["accuracy"],
    },
    "video_mme": {
        "loader": "video_mme",
        "path_candidates": [
            DATASETS_DIR / "video" / "Video-MME_hf",
            DATASETS_DIR / "video" / "Video-MME",
            DATASETS_DIR / "video" / "Video_MME",
            DATASETS_DIR / "Video-MME",
            DATASETS_DIR / "Video_MME",
        ],
        "metrics": ["accuracy"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Input builders
# ═══════════════════════════════════════════════════════════════════

def build_inputs_qwen3vl(processor, image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    skip = {"mm_token_type_ids"}
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items() if k not in skip}


def build_inputs_qwen3vl_video(processor, video_path, prompt):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(video_path)},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    skip = {"mm_token_type_ids"}
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items() if k not in skip}


def build_inputs_llava(processor, image, prompt):
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt")
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


INPUT_BUILDERS = {
    "qwen3vl": build_inputs_qwen3vl,
    "llava": build_inputs_llava,
}


def build_inputs(processor, builder_name, media, prompt):
    if builder_name == "qwen3vl" and isinstance(media, (str, Path)) and str(media).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return build_inputs_qwen3vl_video(processor, media, prompt)
    return INPUT_BUILDERS[builder_name](processor, media, prompt)


# ═══════════════════════════════════════════════════════════════════
# Data loaders -- each yields (image, prompt, ground_truth_dict)
# ═══════════════════════════════════════════════════════════════════

def load_pope(cfg, n):
    with open(cfg["path"]) as f:
        items = [json.loads(line) for line in f]
    items = items[:n]
    image_dir = Path(cfg["image_dir"])
    for item in items:
        img_path = image_dir / item["image"]
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        prompt = item["text"] + " Answer yes or no."
        yield image, prompt, {"label": item["label"].strip().lower()}


def load_chair(cfg, n):
    image_dir = Path(cfg["image_dir"])
    imgs = sorted(image_dir.glob("*.jpg"))[:n]
    for img_path in imgs:
        image = Image.open(img_path).convert("RGB")
        yield image, "Describe this image in detail.", {"image_file": img_path.name}


def _parquet_image(row, image_cols):
    for col in image_cols:
        val = row.get(col)
        if val is None:
            continue
        if isinstance(val, dict) and "bytes" in val:
            return Image.open(BytesIO(val["bytes"])).convert("RGB")
        if isinstance(val, bytes):
            return Image.open(BytesIO(val)).convert("RGB")
    return None


def load_mmbench(cfg, n):
    df = pd.read_parquet(cfg["path"])
    df = df.head(n)
    image_cols = [c for c in df.columns if "image" in c.lower()]
    for _, row in df.iterrows():
        image = _parquet_image(row, image_cols)
        if image is None:
            continue
        q = str(row.get("question", ""))
        hint = str(row.get("hint", ""))
        choices = ""
        for letter in ["A", "B", "C", "D"]:
            val = row.get(letter)
            if val is not None and str(val).strip():
                choices += f"\n{letter}. {val}"
        prompt = f"{hint}\n{q}{choices}\nAnswer with the letter only."
        yield image, prompt, {"answer": str(row.get("answer", "")).strip()}


def load_mme(cfg, n):
    data_dir = Path(cfg["path"])
    files = sorted(data_dir.glob("*.parquet"))
    count = 0
    for pf in files:
        df = pd.read_parquet(pf)
        image_cols = [c for c in df.columns if "image" in c.lower()]
        for _, row in df.iterrows():
            if count >= n:
                return
            image = _parquet_image(row, image_cols)
            if image is None:
                continue
            prompt = str(row.get("question", ""))
            gt = str(row.get("answer", "")).strip().lower()
            cat = str(row.get("category", ""))
            yield image, prompt, {"answer": gt, "category": cat}
            count += 1


def load_hallusionbench(cfg, n):
    df = pd.read_parquet(cfg["path"])
    df = df.head(n)
    image_cols = [c for c in df.columns if "image" in c.lower()]
    for _, row in df.iterrows():
        image = _parquet_image(row, image_cols)
        if image is None:
            continue
        prompt = str(row.get("question", "")) + " Answer yes or no."
        raw_gt = str(row.get("gt_answer", "")).strip()
        gt = {"1": "yes", "0": "no"}.get(raw_gt, raw_gt.lower())
        yield image, prompt, {"answer": gt}


def load_mmmu(cfg, n, manifest_path=None):
    root = Path(cfg["path"])
    manifest_rows = []
    if manifest_path:
        manifest_rows = _load_manifest_rows(Path(manifest_path))

    samples = []
    sample_count_by_subject = defaultdict(int)
    subjects = []

    if manifest_rows:
        df_cache = {}
        for row in manifest_rows:
            if len(samples) >= n:
                break
            parquet_path = Path(row["parquet_path"])
            if not parquet_path.is_absolute():
                parquet_path = root / parquet_path
            if parquet_path not in df_cache:
                df_cache[parquet_path] = pd.read_parquet(parquet_path)
            df = df_cache[parquet_path]
            sample_id = str(row["sample_id"])
            row_index = row.get("row_index")
            if row_index is not None:
                sample_row = df.iloc[int(row_index)]
            else:
                matched = df[df["id"].astype(str) == sample_id]
                if matched.empty:
                    continue
                sample_row = matched.iloc[0]
            subject = str(row.get("subject", parquet_path.parent.name))
            sample = _build_mmmu_sample(sample_row, subject)
            if sample is None:
                continue
            samples.append(sample)
            sample_count_by_subject[subject] += 1
            if subject not in subjects:
                subjects.append(subject)
    else:
        for subject_dir in sorted(root.iterdir()):
            if len(samples) >= n:
                break
            if not subject_dir.is_dir() or subject_dir.name.startswith("."):
                continue
            parquet_files = sorted(subject_dir.glob("validation-*.parquet"))
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                for _, row in df.iterrows():
                    if len(samples) >= n:
                        break
                    sample = _build_mmmu_sample(row, subject_dir.name)
                    if sample is None:
                        continue
                    samples.append(sample)
                    sample_count_by_subject[subject_dir.name] += 1
                    if subject_dir.name not in subjects:
                        subjects.append(subject_dir.name)
                if len(samples) >= n:
                    break

    return samples, {
        "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
        "sample_count_by_subject": dict(sample_count_by_subject),
        "mmmu_subjects": subjects,
    }


def _build_mmmu_sample(row, subject: str):
    image = None
    for i in range(1, 8):
        col = f"image_{i}"
        val = row.get(col)
        if val is not None:
            if isinstance(val, dict) and "bytes" in val:
                image = Image.open(BytesIO(val["bytes"])).convert("RGB")
                break
            if isinstance(val, bytes):
                image = Image.open(BytesIO(val)).convert("RGB")
                break
    if image is None:
        return None
    q = str(row.get("question", ""))
    opts = row.get("options")
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = None
    choice_str = ""
    if isinstance(opts, list):
        for i, o in enumerate(opts):
            letter = chr(65 + i)
            choice_str += f"\n{letter}. {o}"
    prompt = f"{q}{choice_str}\nAnswer with the letter only."
    return image, prompt, {
        "answer": str(row.get("answer", "")).strip(),
        "subject": subject,
        "sample_id": str(row.get("id", "")),
        "question_type": str(row.get("question_type", "")),
    }


def _iter_nonzero_parquets(path: Path):
    for pf in sorted(path.glob("*.parquet")):
        try:
            if pf.stat().st_size > 0:
                yield pf
        except OSError:
            continue


def _resolve_existing_path(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return None


def _load_manifest_rows(manifest_path: Path):
    manifest_path = Path(manifest_path)
    if manifest_path.suffix.lower() == ".csv":
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload.get("entries", [])
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_subject_name(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(subject).strip().lower()).strip("_")


def _normalize_text_value(text) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _compute_anls_score(pred: str, gt: str) -> float:
    pred_norm = _normalize_text_value(pred)
    gt_norm = _normalize_text_value(gt)
    if not pred_norm and not gt_norm:
        return 1.0
    if not pred_norm or not gt_norm:
        return 0.0
    dist = _levenshtein_distance(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm), 1)
    score = 1.0 - (dist / max_len)
    return score if score >= 0.5 else 0.0


def _read_video_mme_index(index_path: Path):
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    by_video_id = {}
    for entry in entries:
        key = str(entry.get("video_id", "")).strip()
        if key:
            by_video_id[key] = entry
    return by_video_id, payload.get("stats", {})


def _parse_video_mme_options(options):
    if hasattr(options, "tolist"):
        options = options.tolist()
    if not isinstance(options, list):
        return [], {}
    pairs = []
    answer_map = {}
    for idx, option in enumerate(options):
        raw = str(option).strip()
        letter = chr(65 + idx)
        match = re.match(r"^\s*([A-D])[\.\):\s-]+(.*)$", raw, flags=re.IGNORECASE)
        option_text = match.group(2).strip() if match else raw
        pairs.append((letter, option_text))
        answer_map[letter] = option_text
    return pairs, answer_map


def _options_to_letters(options):
    if options is None:
        return [], {}
    if hasattr(options, "tolist"):
        options = options.tolist()
    if not isinstance(options, list):
        return [], {}
    pairs = []
    answer_map = {}
    for idx, option in enumerate(options):
        letter = chr(65 + idx)
        option_text = str(option)
        pairs.append((letter, option_text))
        answer_map[option_text.strip().lower()] = letter
    return pairs, answer_map


def load_freak(cfg, n):
    count = 0
    for pf in _iter_nonzero_parquets(Path(cfg["path"])):
        df = pd.read_parquet(pf)
        image_cols = [c for c in df.columns if "image" in c.lower()]
        for _, row in df.iterrows():
            if count >= n:
                return
            image = _parquet_image(row, image_cols)
            if image is None:
                continue
            question = str(row.get("question", "")).strip()
            pairs, answer_map = _options_to_letters(row.get("options"))
            choice_text = "".join(f"\n{letter}. {text}" for letter, text in pairs)
            prompt = f"{question}{choice_text}\nAnswer with the option letter only."
            gt_text = str(row.get("ground_truth", "")).strip()
            gt_letter = answer_map.get(gt_text.lower())
            yield image, prompt, {
                "answer": gt_letter or gt_text,
                "answer_letter": gt_letter,
                "answer_text": gt_text,
                "task_type": "letter_or_text",
                "subset": str(row.get("type", "")),
                "item": str(row.get("item", "")),
            }
            count += 1


def load_docvqa(cfg, n):
    root = _resolve_existing_path(cfg.get("path_candidates", []))
    if root is None:
        return
    parquet_files = [root] if root.is_file() else list(_iter_nonzero_parquets(root))
    count = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        image_cols = [c for c in df.columns if "image" in c.lower()]
        for _, row in df.iterrows():
            if count >= n:
                return
            image = _parquet_image(row, image_cols)
            if image is None:
                continue
            question = str(row.get("question", row.get("query", ""))).strip()
            answer_candidates = []
            for key in ("answer", "answers"):
                value = row.get(key)
                if value is None:
                    continue
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if isinstance(value, list):
                    answer_candidates.extend(str(item).strip() for item in value if str(item).strip())
                else:
                    text = str(value).strip()
                    if text and text.lower() != "none":
                        answer_candidates.append(text)
            answer_text = answer_candidates[0] if answer_candidates else ""
            if not question or not answer_text:
                continue
            prompt = f"{question}\nAnswer briefly using the text in the image when needed."
            yield image, prompt, {
                "answer": answer_text,
                "answer_text": answer_text,
                "task_type": "text",
            }
            count += 1


def _resolve_vidhalluc_video(cfg, video_key: str) -> Optional[Path]:
    data_dir = Path(cfg["data_dir"])
    extracted = list(data_dir.rglob(f"{video_key}.*"))
    for candidate in extracted:
        if candidate.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            return candidate

    prefix = video_key.split("_", 1)[0].lower()
    zip_map = {
        "obb4013eic8": "ACH_videos.zip",
        "vccwvryqu2i": "ACH_videos.zip",
    }
    zip_name = zip_map.get(prefix)
    if zip_name is None:
        if "_clip_" in video_key:
            zip_name = "ACH_videos.zip"
    zip_path = data_dir / zip_name if zip_name else None
    if zip_path is None or not zip_path.exists():
        return None

    TEMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if video_key in member and member.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    out_path = TEMP_CACHE_DIR / Path(member).name
                    if not out_path.exists():
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                    return out_path
    except zipfile.BadZipFile:
        return None
    return None


def load_vidhalluc(cfg, n):
    root = Path(cfg["path"])
    count = 0

    binary_path = root / "ach_binaryqa.json"
    if binary_path.exists():
        data = json.loads(binary_path.read_text(encoding="utf-8"))
        for _, entries in data.items():
            if count >= n:
                return
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if count >= n:
                    return
                question = str(entry.get("q", "")).strip()
                answers = entry.get("a", {})
                if not isinstance(answers, dict):
                    continue
                for video_key, answer in answers.items():
                    video_path = _resolve_vidhalluc_video(cfg, str(video_key))
                    if video_path is None:
                        continue
                    prompt = f"{question} Answer yes or no."
                    yield video_path, prompt, {"answer": str(answer).strip().lower(), "task_type": "yes_no"}
                    count += 1
                    if count >= n:
                        return

    mcq_path = root / "ach_mcq.json"
    if mcq_path.exists():
        data = json.loads(mcq_path.read_text(encoding="utf-8"))
        for _, entries in data.items():
            if count >= n:
                return
            if not isinstance(entries, dict):
                continue
            for video_key, payload in entries.items():
                if count >= n:
                    return
                if not isinstance(payload, dict):
                    continue
                video_path = _resolve_vidhalluc_video(cfg, str(video_key))
                if video_path is None:
                    continue
                question = str(payload.get("Question", "")).strip()
                choices = payload.get("Choices", {})
                choice_text = "".join(f"\n{letter}. {text}" for letter, text in choices.items())
                prompt = f"{question}{choice_text}\nAnswer with the option letter only."
                yield video_path, prompt, {
                    "answer": str(payload.get("Correct Answer", "")).strip(),
                    "task_type": "letter",
                }
                count += 1
                if count >= n:
                    return


def load_video_mme(cfg, n, index_path=None):
    root = _resolve_existing_path(cfg.get("path_candidates", []))
    if root is None:
        return [], {"loader_errors": {"missing_annotation": 0, "missing_video_file": 0, "unresolved_video_id": 0, "bad_json_layout": 1}}

    resolved_index_path = Path(index_path) if index_path else root / "video_mme_index.json"
    index_rows = {}
    index_stats = {}
    if resolved_index_path.exists():
        index_rows, index_stats = _read_video_mme_index(resolved_index_path)

    annotation_root = root / "videomme"
    parquet_files = list(_iter_nonzero_parquets(annotation_root)) if annotation_root.exists() else []
    if not parquet_files and root.is_file() and root.suffix == ".parquet":
        parquet_files = [root]

    errors = defaultdict(int)
    samples = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception:
            errors["bad_json_layout"] += 1
            continue
        for _, row in df.iterrows():
            if len(samples) >= n:
                break
            question = str(row.get("question", row.get("query", ""))).strip()
            answer = str(row.get("answer", row.get("correct_answer", ""))).strip()
            video_id = str(row.get("videoID") or row.get("video_id") or row.get("video") or row.get("video_path") or "").strip()
            if not question or not answer or not video_id:
                errors["missing_annotation"] += 1
                continue

            entry = index_rows.get(video_id)
            if entry is None and row.get("video_id") is not None:
                entry = index_rows.get(str(row.get("video_id")).strip())
            if entry is None:
                errors["unresolved_video_id"] += 1
                continue

            video_path = _materialize_video_mme_video(entry)
            if video_path is None or not video_path.exists():
                errors["missing_video_file"] += 1
                continue

            option_pairs, answer_map = _parse_video_mme_options(row.get("options"))
            prompt = question
            if option_pairs:
                prompt += "".join(f"\n{letter}. {text}" for letter, text in option_pairs)
                prompt += "\nAnswer with the option letter only."
                answer_letter = answer.strip().upper()
                answer_text = answer_map.get(answer_letter, answer)
                task_type = "letter"
            else:
                prompt += "\nAnswer briefly."
                answer_letter = None
                answer_text = answer
                task_type = "text"

            samples.append((
                video_path,
                prompt,
                {
                    "answer": answer,
                    "answer_letter": answer_letter,
                    "answer_text": answer_text,
                    "task_type": task_type,
                    "video_id": video_id,
                },
            ))

    return samples, {
        "video_mme_index_path": str(resolved_index_path.resolve()) if resolved_index_path.exists() else None,
        "video_mme_index_stats": index_stats,
        "loader_errors": {
            "missing_annotation": int(errors["missing_annotation"]),
            "missing_video_file": int(errors["missing_video_file"]),
            "unresolved_video_id": int(errors["unresolved_video_id"]),
            "bad_json_layout": int(errors["bad_json_layout"]),
        },
    }


def _materialize_video_mme_video(index_entry: dict) -> Optional[Path]:
    resolved_path = Path(index_entry["resolved_video_path"])
    if resolved_path.exists():
        return resolved_path

    source_path = Path(index_entry["source_chunk_or_zip"])
    if not source_path.exists():
        return None
    if source_path.suffix.lower() != ".zip":
        return source_path if source_path.exists() else None

    member_name = index_entry.get("zip_member")
    if not member_name:
        return None
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source_path) as zf:
            with zf.open(member_name) as src, open(resolved_path, "wb") as dst:
                dst.write(src.read())
        return resolved_path
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


LOADERS = {
    "pope": load_pope,
    "chair": load_chair,
    "mmbench": load_mmbench,
    "mme": load_mme,
    "hallusionbench": load_hallusionbench,
    "mmmu": load_mmmu,
    "freak": load_freak,
    "docvqa": load_docvqa,
    "vidhalluc": load_vidhalluc,
    "video_mme": load_video_mme,
}


# ═══════════════════════════════════════════════════════════════════
# Answer extraction helpers
# ═══════════════════════════════════════════════════════════════════

def extract_yes_no(text):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    words = re.findall(r"\b(yes|no)\b", t)
    return words[0] if words else "unknown"


def extract_letter(text):
    t = text.strip()
    m = re.match(r"^([A-D])", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1)
    return text.strip()[:1].upper()


# ═══════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════

def compute_pope_metrics(predictions, ground_truths):
    tp = fp = tn = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        p = extract_yes_no(pred)
        g = gt["label"]
        if p == "yes" and g == "yes":
            tp += 1
        elif p == "yes" and g == "no":
            fp += 1
        elif p == "no" and g == "no":
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_chair_metrics(predictions, ground_truths, ann_path):
    with open(ann_path) as f:
        instances = json.load(f)
    coco_objects = set()
    for cat in instances["categories"]:
        coco_objects.add(cat["name"].lower())

    synonyms = {
        "tv": "television", "couch": "sofa", "cell phone": "mobile phone",
        "airplane": "aeroplane", "handbag": "purse",
    }
    for k, v in list(synonyms.items()):
        synonyms[v] = k
    all_names = set(coco_objects)
    for k, v in synonyms.items():
        all_names.add(k)
        all_names.add(v)

    image_id_to_anns = defaultdict(set)
    for ann in instances["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        cat_name = None
        for c in instances["categories"]:
            if c["id"] == cat_id:
                cat_name = c["name"].lower()
                break
        if cat_name:
            image_id_to_anns[img_id].add(cat_name)

    hall_sentences = 0
    total_sentences = 0
    hall_objects = 0
    total_objects = 0
    gen_lengths = []

    for pred, gt in zip(predictions, ground_truths):
        img_file = gt["image_file"]
        img_id_match = re.search(r"_(\d+)\.jpg", img_file)
        img_id = int(img_id_match.group(1)) if img_id_match else 0
        gt_objects = image_id_to_anns.get(img_id, set())

        gen_lengths.append(len(pred.split()))

        sentences = re.split(r"[.!?]+", pred)
        for sent in sentences:
            sent_lower = sent.lower()
            if not sent_lower.strip():
                continue
            total_sentences += 1
            sent_has_hall = False
            for obj_name in all_names:
                pattern = r"\b" + re.escape(obj_name) + r"\b"
                if re.search(pattern, sent_lower):
                    total_objects += 1
                    canonical = synonyms.get(obj_name, obj_name)
                    if canonical not in gt_objects and obj_name not in gt_objects:
                        hall_objects += 1
                        sent_has_hall = True
            if sent_has_hall:
                hall_sentences += 1

    chair_s = hall_sentences / total_sentences if total_sentences else 0
    chair_i = hall_objects / total_objects if total_objects else 0
    agl = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0

    return {"chair_s": chair_s, "chair_i": chair_i, "agl": agl}


def compute_accuracy_metrics(predictions, ground_truths, answer_type="letter"):
    correct = 0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truths):
        gt_ans = gt["answer"].strip().lower()
        if answer_type == "yes_no":
            pred_ans = extract_yes_no(pred)
        else:
            pred_ans = extract_letter(pred).lower()
            gt_ans = gt_ans.lower()
        if pred_ans == gt_ans:
            correct += 1
    return {"accuracy": correct / total if total else 0}


def compute_mixed_accuracy_metrics(predictions, ground_truths):
    correct = 0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truths):
        task_type = gt.get("task_type", "letter")
        if task_type == "yes_no":
            pred_ans = extract_yes_no(pred)
            gt_ans = str(gt.get("answer", "")).strip().lower()
            correct += int(pred_ans == gt_ans)
            continue

        pred_letter = extract_letter(pred).upper()
        gt_letter = str(gt.get("answer_letter") or gt.get("answer", "")).strip().upper()
        gt_text = str(gt.get("answer_text") or gt.get("answer", "")).strip().lower()
        pred_text = pred.strip().lower()
        if pred_letter == gt_letter or gt_text in pred_text:
            correct += 1
    return {"accuracy": correct / total if total else 0}


def compute_freak_leakage_metrics(predictions, ground_truths, calibrator_catalog_path: str | None):
    if not calibrator_catalog_path:
        return None, {}

    catalog_entities = load_catalog_entities(calibrator_catalog_path)
    benchmark_entities = build_freak_entity_index(ground_truths)
    summary = build_leakage_summary(catalog_entities, benchmark_entities)
    status_by_category = {
        row["category"]: row["status"] for row in summary["per_category_summary"]
    }

    seen_preds, seen_gts = [], []
    unseen_preds, unseen_gts = [], []
    for pred, gt in zip(predictions, ground_truths):
        category = str(gt.get("item") or gt.get("subset") or gt.get("answer_text") or "").strip()
        status = status_by_category.get(category, "unseen")
        if status == "seen":
            seen_preds.append(pred)
            seen_gts.append(gt)
        else:
            unseen_preds.append(pred)
            unseen_gts.append(gt)

    out = {
        "seen_split_accuracy": compute_mixed_accuracy_metrics(seen_preds, seen_gts).get("accuracy", 0.0) if seen_gts else 0.0,
        "unseen_split_accuracy": compute_mixed_accuracy_metrics(unseen_preds, unseen_gts).get("accuracy", 0.0) if unseen_gts else 0.0,
        "seen_sample_count": len(seen_gts),
        "unseen_sample_count": len(unseen_gts),
    }
    summary["catalog_path"] = str(Path(calibrator_catalog_path).resolve())
    return summary, out


def compute_text_match_metrics(predictions, ground_truths):
    correct = 0
    total = 0
    normalized_exact = 0
    option_letter_correct = 0
    anls_scores = []

    for pred, gt in zip(predictions, ground_truths):
        gt_text = str(gt.get("answer_text", gt.get("answer", ""))).strip()
        if not gt_text:
            continue
        total += 1
        pred_norm = _normalize_text_value(pred)
        gt_norm = _normalize_text_value(gt_text)
        exact_match = int(pred_norm == gt_norm)
        normalized_exact += exact_match

        task_type = gt.get("task_type", "text")
        matched = exact_match
        if task_type == "letter":
            pred_letter = extract_letter(pred).upper()
            gt_letter = str(gt.get("answer_letter") or gt.get("answer", "")).strip().upper()
            letter_match = int(bool(gt_letter) and pred_letter == gt_letter)
            option_letter_correct += letter_match
            matched = bool(letter_match or exact_match)
        else:
            anls_scores.append(_compute_anls_score(pred, gt_text))
            matched = exact_match

        correct += int(matched)

    metrics = {
        "accuracy": correct / total if total else 0,
        "normalized_exact_match": normalized_exact / total if total else 0,
        "anls": sum(anls_scores) / len(anls_scores) if anls_scores else 0.0,
    }
    if option_letter_correct or any(gt.get("task_type") == "letter" for gt in ground_truths):
        metrics["option_letter_match"] = option_letter_correct / total if total else 0
    return metrics


def compute_mme_metrics(predictions, ground_truths):
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    for pred, gt in zip(predictions, ground_truths):
        cat = gt.get("category", "unknown")
        gt_ans = gt["answer"].strip().lower()
        pred_ans = extract_yes_no(pred)
        cat_total[cat] += 1
        if pred_ans == gt_ans:
            cat_correct[cat] += 1
    total_score = 0
    for cat in cat_total:
        acc = cat_correct[cat] / cat_total[cat] if cat_total[cat] else 0
        total_score += acc * 100
    overall_correct = sum(cat_correct.values())
    overall_total = sum(cat_total.values())
    return {
        "perception_score": total_score,
        "accuracy": overall_correct / overall_total if overall_total else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Core evaluation loop
# ═══════════════════════════════════════════════════════════════════

def run_single_inference(model, processor, builder_name, media, prompt, max_tokens=100):
    inputs = build_inputs(processor, builder_name, media, prompt)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens)
    elapsed = time.time() - t0
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return text, len(gen_ids), elapsed


def run_bra_inference(model, processor, builder_name, adapter, media, prompt,
                      bra_config, tokenizer, max_tokens=100):
    from bra_logits_processor import create_bra_processor
    inputs = build_inputs(processor, builder_name, media, prompt)
    video_grid_thw = inputs.get("video_grid_thw")
    extractor, bra_proc = create_bra_processor(
        model, adapter, inputs["input_ids"], config=bra_config,
        tokenizer=tokenizer, video_grid_thw=video_grid_thw)
    try:
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 logits_processor=[bra_proc])
        elapsed = time.time() - t0
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        return text, len(gen_ids), elapsed, bra_proc.get_stats(), bra_proc.get_audit_log()
    finally:
        extractor.remove()
        bra_proc.reset()


def evaluate_dataset(model, processor, builder_name, adapter, ds_name, ds_cfg,
                     n_samples, bra_config, tokenizer, args):
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {ds_name} (N={n_samples})")
    print(f"{'─'*60}")

    loader_type = ds_cfg["loader"]
    loader_meta = {}
    if loader_type == "mmmu":
        samples, loader_meta = load_mmmu(ds_cfg, n_samples, manifest_path=args.mmmu_manifest)
    elif loader_type == "video_mme":
        samples, loader_meta = load_video_mme(ds_cfg, n_samples, index_path=args.video_mme_index)
    else:
        loader_fn = LOADERS[loader_type]
        samples = list(loader_fn(ds_cfg, n_samples))
    actual_n = len(samples)
    print(f"  Loaded {actual_n} samples")

    if actual_n == 0:
        return None

    base_preds = []
    bra_preds = []
    gts = []
    bra_stats = []
    sample_audits = []
    sample_errors = []
    base_lengths, bra_lengths = [], []
    base_latencies, bra_latencies = [], []
    base_peak_vram_bytes = 0
    bra_peak_vram_bytes = 0

    for i, (media, prompt, gt) in enumerate(samples):
        t0 = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            base_text, base_len, base_elapsed = run_single_inference(model, processor, builder_name, media, prompt)
            base_peak_vram_bytes = max(base_peak_vram_bytes, int(torch.cuda.max_memory_allocated()))
            base_preds.append(base_text)
            base_lengths.append(base_len)
            if base_len > 0:
                base_latencies.append((base_elapsed * 1000.0) / base_len)

            torch.cuda.reset_peak_memory_stats()
            bra_text, bra_len, bra_elapsed, proc_stats, proc_audit = run_bra_inference(
                model, processor, builder_name, adapter, media, prompt, bra_config, tokenizer
            )
            bra_peak_vram_bytes = max(bra_peak_vram_bytes, int(torch.cuda.max_memory_allocated()))
            bra_preds.append(bra_text)
            bra_lengths.append(bra_len)
            if bra_len > 0:
                bra_latencies.append((bra_elapsed * 1000.0) / bra_len)
            bra_stats.append(proc_stats)
            gts.append(gt)
            if proc_audit and len(sample_audits) < 5:
                sample_audits.append({
                    "sample_index": i,
                    "ground_truth": gt,
                    "base_preview": base_text[:160],
                    "bra_preview": bra_text[:160],
                    "audit": proc_audit[:5],
                })

            elapsed = time.time() - t0
            if i < 3 or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{actual_n}] {elapsed:.1f}s  "
                      f"base='{base_text[:40]}...'  bra='{bra_text[:40]}...'")
        except Exception as exc:
            sample_errors.append({
                "sample_index": i,
                "ground_truth": gt,
                "error": str(exc),
            })
            continue

    if loader_type == "pope":
        base_metrics = compute_pope_metrics(base_preds, gts)
        bra_metrics = compute_pope_metrics(bra_preds, gts)
    elif loader_type == "chair":
        base_metrics = compute_chair_metrics(base_preds, gts, ds_cfg["ann_path"])
        bra_metrics = compute_chair_metrics(bra_preds, gts, ds_cfg["ann_path"])
    elif loader_type == "mme":
        base_metrics = compute_mme_metrics(base_preds, gts)
        bra_metrics = compute_mme_metrics(bra_preds, gts)
    elif loader_type == "hallusionbench":
        base_metrics = compute_accuracy_metrics(base_preds, gts, "yes_no")
        bra_metrics = compute_accuracy_metrics(bra_preds, gts, "yes_no")
    elif loader_type in {"freak", "vidhalluc"}:
        base_metrics = compute_mixed_accuracy_metrics(base_preds, gts)
        bra_metrics = compute_mixed_accuracy_metrics(bra_preds, gts)
        if loader_type == "freak":
            leakage_summary, base_leakage = compute_freak_leakage_metrics(base_preds, gts, args.calibrator_catalog)
            _, bra_leakage = compute_freak_leakage_metrics(bra_preds, gts, args.calibrator_catalog)
            if leakage_summary:
                loader_meta["category_leakage_audit"] = leakage_summary
            base_metrics.update(base_leakage)
            bra_metrics.update(bra_leakage)
    elif loader_type in {"docvqa", "video_mme"}:
        base_metrics = compute_text_match_metrics(base_preds, gts)
        bra_metrics = compute_text_match_metrics(bra_preds, gts)
    else:
        base_metrics = compute_accuracy_metrics(base_preds, gts, "letter")
        bra_metrics = compute_accuracy_metrics(bra_preds, gts, "letter")

    base_metrics.update({
        "agl": sum(base_lengths) / max(len(base_lengths), 1),
        "agl_stddev": float(pd.Series(base_lengths, dtype="float64").std(ddof=0)) if base_lengths else 0.0,
        "itl_ms_per_token": sum(base_latencies) / max(len(base_latencies), 1),
        "tpot_ms_per_token": sum(base_latencies) / max(len(base_latencies), 1),
        "tokens_per_second": 1000.0 / max(sum(base_latencies) / max(len(base_latencies), 1), 1e-6),
        "peak_vram_gb": base_peak_vram_bytes / 1e9,
    })
    bra_metrics.update(aggregate_bra_stats(bra_stats))
    bra_metrics.update({
        "agl": sum(bra_lengths) / max(len(bra_lengths), 1),
        "agl_stddev": float(pd.Series(bra_lengths, dtype="float64").std(ddof=0)) if bra_lengths else 0.0,
        "itl_ms_per_token": sum(bra_latencies) / max(len(bra_latencies), 1),
        "tpot_ms_per_token": sum(bra_latencies) / max(len(bra_latencies), 1),
        "tokens_per_second": 1000.0 / max(sum(bra_latencies) / max(len(bra_latencies), 1), 1e-6),
        "peak_vram_gb": bra_peak_vram_bytes / 1e9,
    })
    base_itl = base_metrics.get("itl_ms_per_token", 0.0)
    bra_itl = bra_metrics.get("itl_ms_per_token", 0.0)
    base_metrics["latency_multiplier_vs_base"] = 1.0
    bra_metrics["latency_multiplier_vs_base"] = (bra_itl / base_itl) if base_itl else (1.0 if bra_itl == 0 else None)
    notes = build_notes(ds_name, base_metrics, bra_metrics)
    return {
        "dataset": ds_name,
        "n_samples": actual_n,
        "baseline": base_metrics,
        "bra": bra_metrics,
        "bra_method": bra_config.mode,
        "sample_count": actual_n,
        "model_family": getattr(adapter, "name", "unknown"),
        "notes": notes,
        "sample_audits": sample_audits,
        "sample_errors": sample_errors[:10],
        **loader_meta,
    }


def aggregate_bra_stats(stats_list):
    if not stats_list:
        return {}
    keys = [
        "avg_candidate_window",
        "avg_visual_topk",
        "avg_resonance_time_ms",
        "avg_routing_time_ms",
        "avg_vasm_time_ms",
        "intervention_rate",
        "continuation_success_rate",
    ]
    out = {}
    for key in keys:
        vals = [float(s.get(key, 0.0)) for s in stats_list]
        out[key] = sum(vals) / max(len(vals), 1)
    frame_hist = defaultdict(int)
    for s in stats_list:
        for k, v in s.get("selected_frame_histogram", {}).items():
            frame_hist[int(k)] += int(v)
    if frame_hist:
        out["selected_frame_histogram"] = dict(sorted(frame_hist.items()))
    for key in ("visual_state_provenance", "vasm_metadata"):
        for s in stats_list:
            if s.get(key):
                out[key] = s[key]
                break
    failure_examples = []
    for s in stats_list:
        failure_examples.extend(s.get("continuation_failure_examples", []))
    out["continuation_failure_examples"] = failure_examples[:10]
    out["suffix_collapse_failures"] = int(sum(int(s.get("suffix_collapse_failures", 0)) for s in stats_list))
    out["continuation_attempts"] = int(sum(int(s.get("continuation_attempts", 0)) for s in stats_list))
    return out


def build_notes(dataset_name: str, baseline_metrics: dict, bra_metrics: dict):
    notes = []
    if dataset_name == "chair" and bra_metrics.get("agl", 0.0) >= 230:
        notes.append("chair_agl_near_cap")
    if "intervention_rate" in bra_metrics and bra_metrics["intervention_rate"] >= 0.95:
        notes.append("high_trigger_rate")
    if dataset_name in {"vidhalluc", "video_mme"} and not bra_metrics.get("selected_frame_histogram"):
        notes.append("no_temporal_histogram")
    return notes


def build_eval_bra_config(method_name: str, args):
    from bra_logits_processor import make_bra_config

    if method_name in {"bra_zero"}:
        return make_bra_config("bra_zero", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name in {"tlra", "tlra_zero"}:
        return make_bra_config("tlra_zero", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name in {"bra_calib", "tlra_calib"}:
        return make_bra_config(
            method_name,
            warmup_steps=3,
            vasm_artifact_path=args.vasm_artifact,
            projector_checkpoint=args.projector_checkpoint,
        )
    if method_name in {"tlra_full", "tlra_adaptivetopk", "tlra_randomk", "tlra_meanpool", "tlra_max", "tlra_no_vasm"}:
        return make_bra_config(
            method_name,
            warmup_steps=3,
            vasm_artifact_path=args.vasm_artifact,
            projector_checkpoint=args.projector_checkpoint,
        )
    if method_name in {"ablation_meanpool", "bra_meanpool"}:
        return make_bra_config("ablation_meanpool", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name in {"ablation_maxpool", "bra_maxpool", "bra_max"}:
        return make_bra_config("ablation_maxpool", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name in {"bra_v1", "bra_v1_like", "tlra_v1_like"}:
        return make_bra_config(method_name, warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name in {"bra_no_vasm"}:
        return make_bra_config("bra_no_vasm", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name == "legacy_v2":
        return make_bra_config("legacy_v2", warmup_steps=3, vasm_artifact_path=args.vasm_artifact)
    if method_name == "binary_mask":
        return make_bra_config("bra_zero", warmup_steps=3, mask_variant="binary_mask", vasm_artifact_path=args.vasm_artifact)
    if method_name == "entropy_mask":
        return make_bra_config("bra_zero", warmup_steps=3, mask_variant="entropy_mask", vasm_artifact_path=args.vasm_artifact)
    if method_name == "no_mask":
        return make_bra_config("bra_zero", warmup_steps=3, mask_variant="no_mask", vasm_artifact_path=args.vasm_artifact)
    raise ValueError(f"Unsupported BRA eval method: {method_name}")


# ═══════════════════════════════════════════════════════════════════
# Report formatting
# ═══════════════════════════════════════════════════════════════════

def print_report(all_results, model_name):
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT: {model_name}")
    print("=" * 70)

    for r in all_results:
        if r is None:
            continue
        ds = r["dataset"]
        n = r["n_samples"]
        print(f"\n  === {ds} (N={n}, bra={r.get('bra_method', 'bra_zero')}) ===")
        print(f"  {'Metric':<20s} {'Baseline':>10s} {'BRA':>10s} {'Delta':>12s}")
        print(f"  {'─'*54}")

        bm = r["baseline"]
        brm = r["bra"]
        for key in bm:
            bv = bm[key]
            brv = brm[key]
            delta = brv - bv
            if "chair" in key:
                pct = f" ({delta/bv*100:+.1f}%)" if bv != 0 else ""
                print(f"  {key:<20s} {bv:>10.4f} {brv:>10.4f} {delta:>+10.4f}{pct}")
            elif "score" in key:
                print(f"  {key:<20s} {bv:>10.1f} {brv:>10.1f} {delta:>+10.1f}")
            elif key == "agl":
                print(f"  {key:<20s} {bv:>10.1f} {brv:>10.1f} {delta:>+10.1f}")
            else:
                pct = f" ({delta*100:+.1f}pp)" if bv != 0 else ""
                print(f"  {key:<20s} {bv:>10.4f} {brv:>10.4f} {delta:>+10.4f}{pct}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--model", type=str, default="all",
                        help="Model key or 'all'")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset key or 'all'")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument(
        "--bra_method",
        type=str,
        default="bra_zero",
        choices=[
            "bra_zero", "bra_calib", "ablation_meanpool", "ablation_maxpool",
            "bra_meanpool", "bra_maxpool", "bra_max", "bra_no_vasm",
            "bra_v1", "bra_v1_like", "legacy_v2", "binary_mask", "entropy_mask", "no_mask",
            "tlra", "tlra_zero", "tlra_full", "tlra_adaptivetopk", "tlra_calib",
            "tlra_meanpool", "tlra_max", "tlra_no_vasm", "tlra_v1_like", "tlra_randomk",
        ],
    )
    parser.add_argument("--vasm_artifact", type=str, default=None)
    parser.add_argument("--projector_checkpoint", type=str, default=None)
    parser.add_argument("--mmmu_manifest", type=str, default=None)
    parser.add_argument("--video_mme_index", type=str, default=None)
    parser.add_argument("--calibrator_catalog", type=str, default=None)
    parser.add_argument("--output", type=str, default="bra_eval_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("BRA EVALUATION MATRIX")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    import transformers
    print(f"Transformers: {transformers.__version__}")
    print(f"N samples: {args.n_samples}")
    print("=" * 70)

    from bra_logits_processor import make_bra_config
    from bra_operator_multi import detect_adapter
    bra_config = build_eval_bra_config(args.bra_method, args)

    models_to_test = (
        {k: MODEL_REGISTRY[k] for k in MODEL_REGISTRY if k == args.model}
        if args.model != "all"
        else MODEL_REGISTRY
    )
    datasets_to_test = (
        {k: DATASET_REGISTRY[k] for k in DATASET_REGISTRY if k == args.dataset}
        if args.dataset != "all"
        else DATASET_REGISTRY
    )

    all_results = {}

    for model_key, mcfg in models_to_test.items():
        model_path = mcfg["path"]
        if not model_path.exists():
            print(f"\n  SKIP {model_key}: path not found")
            continue

        print(f"\n{'═'*70}")
        print(f"  Loading model: {model_key}")
        print(f"{'═'*70}")

        t0 = time.time()
        model_cls = getattr(transformers, mcfg["hf_class"])
        proc_cls = getattr(transformers, mcfg["hf_processor"])
        model = model_cls.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16)
        model = model.cuda().eval()
        processor = proc_cls.from_pretrained(str(model_path))
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        adapter = detect_adapter(model)
        tokenizer = getattr(processor, "tokenizer", processor)
        builder_name = mcfg["builder"]
        print(f"  Adapter: {adapter.name}")

        model_results = []

        for ds_key, ds_cfg in datasets_to_test.items():
            if ds_cfg["loader"] in ("video_json",) and not mcfg.get("supports_video"):
                print(f"\n  SKIP {ds_key}: model does not support video")
                continue

            try:
                result = evaluate_dataset(
                    model, processor, builder_name, adapter,
                    ds_key, ds_cfg, args.n_samples, bra_config, tokenizer, args)
                if result:
                    model_results.append(result)
            except Exception as e:
                print(f"\n  ERROR on {ds_key}: {e}")
                traceback.print_exc()

        print_report(model_results, model_key)
        all_results[model_key] = model_results

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    output_path = PROJECT / args.output
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
