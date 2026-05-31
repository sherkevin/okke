from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO JSONL detector server for CHORD.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_path).to(args.device)
    model.eval()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__EXIT__":
            break
        request = json.loads(line)
        image = Image.open(request["image_path"]).convert("RGB")
        query = request["query"].strip().lower()
        if query and not query.endswith("."):
            query = f"{query}."

        inputs = processor(images=image, text=query, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=args.device)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=float(request["box_threshold"]),
            text_threshold=float(request["text_threshold"]),
            target_sizes=target_sizes,
        )[0]

        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"]
        anchors = []
        for box, score, label in zip(boxes, scores, labels):
            anchors.append(
                {
                    "box": [float(x) for x in box],
                    "confidence": float(score),
                    "phrase": str(label),
                }
            )
        anchors.sort(key=lambda item: item["confidence"], reverse=True)
        payload = {"anchors": anchors[: int(request["max_boxes"])]}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
