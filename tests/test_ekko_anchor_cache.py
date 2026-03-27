from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKKO"))

from chord.knowledge_kernel_evaluator import DetectedAnchor, build_knowledge_kernel_result_from_cache  # noqa: E402
from chord.anchor_cache import AnchorCache, CachedAnchorEntry, build_anchor_cache_key, dump_anchor_cache  # noqa: E402


class EkkoAnchorCacheTests(unittest.TestCase):
    def test_build_anchor_cache_key_normalizes_path_and_rounds_thresholds(self) -> None:
        key = build_anchor_cache_key(
            image_path=r"D:\dataset\coco\0001.jpg",
            query="Is there a car?",
            anchor_query="car",
            box_threshold=0.2500000001,
            text_threshold=0.2000000001,
            max_boxes=8,
        )
        payload = json.loads(key)
        self.assertEqual(payload["image_path"], "D:/dataset/coco/0001.jpg")
        self.assertEqual(payload["anchor_query"], "car")
        self.assertEqual(payload["box_threshold"], 0.25)
        self.assertEqual(payload["text_threshold"], 0.2)

    def test_round_trip_jsonl_cache_lookup_returns_entry(self) -> None:
        entry = CachedAnchorEntry(
            image_path="datasets/coco/0001.jpg",
            image_size=(480, 640),
            query="Is there a car?",
            anchor_query="car",
            box_threshold=0.25,
            text_threshold=0.2,
            max_boxes=8,
            anchors=[DetectedAnchor(box=(1.0, 2.0, 3.0, 4.0), confidence=0.9, phrase="car")],
            grid_size=(1, 1),
            membership=[[1.0]],
            relevance=[1.0],
            confidence=[0.9],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "anchors.jsonl"
            dump_anchor_cache([entry], path)
            cache = AnchorCache.from_jsonl(path)

        cached = cache.get(
            image_path="datasets/coco/0001.jpg",
            query="Is there a car?",
            anchor_query="car",
            box_threshold=0.25,
            text_threshold=0.2,
            max_boxes=8,
        )
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(cached.image_size, (480, 640))
        self.assertEqual(cached.anchor_query, "car")
        self.assertEqual(cached.anchors[0].phrase, "car")

    def test_cached_kernel_payload_can_directly_build_token_weights(self) -> None:
        entry = CachedAnchorEntry(
            image_path="datasets/coco/0001.jpg",
            image_size=(480, 640),
            query="Is there a car?",
            anchor_query="car",
            box_threshold=0.25,
            text_threshold=0.2,
            max_boxes=8,
            anchors=[DetectedAnchor(box=(1.0, 2.0, 3.0, 4.0), confidence=0.9, phrase="car")],
            grid_size=(1, 2),
            membership=[[1.0, 0.0]],
            relevance=[1.0],
            confidence=[0.9],
        )

        result = build_knowledge_kernel_result_from_cache(cached_entry=entry, alpha_anchor=0.5)

        self.assertEqual(result.membership.tolist(), [[1.0, 0.0]])
        self.assertEqual(result.relevance.tolist(), [1.0])
        self.assertTrue(torch.allclose(result.token_weights, torch.tensor([1.45, 1.0], dtype=torch.float32)))

    def test_empty_membership_payload_recovers_expected_rank_two_shape(self) -> None:
        entry = CachedAnchorEntry(
            image_path="datasets/coco/0002.jpg",
            image_size=(480, 640),
            query="Is there a train?",
            anchor_query="train",
            box_threshold=0.25,
            text_threshold=0.2,
            max_boxes=8,
            anchors=[],
            grid_size=(2, 3),
            membership=[],
            relevance=[],
            confidence=[],
        )

        membership = entry.membership_tensor()

        self.assertEqual(tuple(membership.shape), (0, 6))


if __name__ == "__main__":
    unittest.main()
