from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKKO"))

from chord.knowledge_kernel_evaluator import (  # noqa: E402
    DetectedAnchor,
    boxes_to_visual_membership,
    build_anchor_weight_result,
    build_knowledge_kernel_result,
    build_visual_token_weights,
)


class EkkoAnchorBuilderTests(unittest.TestCase):
    def test_membership_maps_boxes_to_grid_centers(self) -> None:
        anchors = [
            DetectedAnchor(box=(0.0, 0.0, 50.0, 50.0), confidence=0.9, phrase="snowboard"),
            DetectedAnchor(box=(50.0, 50.0, 100.0, 100.0), confidence=0.8, phrase="car"),
        ]

        membership = boxes_to_visual_membership(
            anchors,
            image_size=(100, 100),
            grid_size=(2, 2),
        )

        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(membership, expected))

    def test_enhance_only_weights_leave_unmatched_tokens_at_one(self) -> None:
        weights = build_visual_token_weights(
            membership=torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            relevance=torch.tensor([1.0, 0.0], dtype=torch.float32),
            confidence=torch.tensor([0.8, 0.9], dtype=torch.float32),
            alpha_anchor=0.5,
        )

        expected = torch.tensor([1.4, 1.0, 1.4, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(weights, expected))

    def test_query_conditioned_anchor_weight_result_uses_overlap_relevance(self) -> None:
        anchors = [
            DetectedAnchor(box=(0.0, 0.0, 50.0, 50.0), confidence=0.8, phrase="snowboard"),
            DetectedAnchor(box=(50.0, 0.0, 100.0, 50.0), confidence=0.9, phrase="car"),
        ]

        result = build_anchor_weight_result(
            anchors=anchors,
            query="Is there a snowboard in the image?",
            image_size=(100, 100),
            grid_size=(2, 2),
            alpha_anchor=0.5,
        )

        expected_weights = torch.tensor([1.4, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.token_weights, expected_weights))
        self.assertFalse(result.used_fallback)

    def test_knowledge_kernel_result_alias_matches_anchor_builder_behavior(self) -> None:
        anchors = [DetectedAnchor(box=(0.0, 0.0, 50.0, 50.0), confidence=0.8, phrase="snowboard")]

        result = build_knowledge_kernel_result(
            anchors=anchors,
            query="snowboard",
            image_size=(100, 100),
            grid_size=(2, 2),
            alpha_anchor=0.5,
        )

        expected_weights = torch.tensor([1.4, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(result.token_weights, expected_weights))


if __name__ == "__main__":
    unittest.main()
