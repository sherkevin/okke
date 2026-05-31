import unittest

import torch

from chord.anchor_builder import (
    DetectedAnchor,
    boxes_to_visual_membership,
    build_anchor_weight_result,
    build_visual_token_weights,
)


class AnchorBuilderTests(unittest.TestCase):
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

    def test_query_miss_keeps_uniform_weights_but_preserves_membership(self) -> None:
        anchors = [
            DetectedAnchor(box=(0.0, 0.0, 50.0, 50.0), confidence=0.8, phrase="dog"),
        ]

        result = build_anchor_weight_result(
            anchors=anchors,
            query="Is there a snowboard in the image?",
            image_size=(100, 100),
            grid_size=(2, 2),
            alpha_anchor=0.5,
        )

        self.assertTrue(torch.equal(result.token_weights, torch.ones(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(result.membership, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)))
        self.assertTrue(result.used_fallback)


if __name__ == "__main__":
    unittest.main()
import unittest

import torch

from chord.anchor_builder import (
    DetectedAnchor,
    boxes_to_visual_membership,
    build_anchor_weight_result,
    build_visual_token_weights,
)


class AnchorBuilderTests(unittest.TestCase):
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

    def test_query_miss_keeps_uniform_weights_but_preserves_membership(self) -> None:
        anchors = [
            DetectedAnchor(box=(0.0, 0.0, 50.0, 50.0), confidence=0.8, phrase="dog"),
        ]

        result = build_anchor_weight_result(
            anchors=anchors,
            query="Is there a snowboard in the image?",
            image_size=(100, 100),
            grid_size=(2, 2),
            alpha_anchor=0.5,
        )

        self.assertTrue(torch.equal(result.token_weights, torch.ones(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(result.membership, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)))
        self.assertTrue(result.used_fallback)


if __name__ == "__main__":
    unittest.main()
