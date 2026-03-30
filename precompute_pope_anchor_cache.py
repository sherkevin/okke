#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from chord.anchor_cache import CachedAnchorEntry, dump_anchor_cache
from chord.detector_client import GroundingDinoClient
from chord.knowledge_kernel_evaluator import boxes_to_visual_membership, score_anchor_relevance
from chord.query_formulation import extract_anchor_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute detector anchors for POPE samples before CHORD decoding.")
    parser.add_argument("--pope-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--grounding-dino-path", required=True)
    parser.add_argument("--detector-python", required=True)
    parser.add_argument("--detector-device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--grid-height", type=int, default=24)
    parser.add_argument("--grid-width", type=int, default=24)
    parser.add_argument("--anchor-query-mode", type=str, default="raw")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_pope_records(pope_path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    for idx, line in enumerate(pope_path.read_text(encoding="utf-8").splitlines()):
        if limit is not None and idx >= limit:
            break
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def main() -> int:
    args = parse_args()
    pope_path = Path(args.pope_path)
    data_path = Path(args.data_path)
    entries: list[CachedAnchorEntry] = []

    with GroundingDinoClient(
        python_executable=args.detector_python,
        server_script=str(Path(__file__).resolve().parent / "chord" / "detector_server.py"),
        model_path=args.grounding_dino_path,
        device=args.detector_device,
    ) as client:
        for idx, record in enumerate(load_pope_records(pope_path, args.limit), start=1):
            image_path = data_path / record["image"]
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            raw_query = str(record["text"])
            anchor_query = extract_anchor_query(raw_query, mode=args.anchor_query_mode)
            anchors = client.detect(
                image_path=str(image_path),
                query=anchor_query,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                max_boxes=args.max_boxes,
            )
            grid_size = (args.grid_height, args.grid_width)
            membership = boxes_to_visual_membership(
                anchors,
                image_size=(height, width),
                grid_size=grid_size,
            )
            relevance = [score_anchor_relevance(anchor_query, anchor.phrase) for anchor in anchors]
            confidence = [anchor.confidence for anchor in anchors]
            entries.append(
                CachedAnchorEntry(
                    image_path=str(image_path).replace("\\", "/"),
                    image_size=(height, width),
                    query=raw_query,
                    anchor_query=anchor_query,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    max_boxes=args.max_boxes,
                    anchors=anchors,
                    grid_size=grid_size,
                    membership=membership.tolist(),
                    relevance=relevance,
                    confidence=confidence,
                )
            )
            if idx % 50 == 0:
                print(f"[knowledge-kernel-cache] processed {idx} samples")

    dump_anchor_cache(entries, args.output_jsonl)
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "entries": len(entries),
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "max_boxes": args.max_boxes,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
