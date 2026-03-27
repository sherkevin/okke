from __future__ import annotations

import unittest

import torch

from inspect_uniground_v2_payload import inspect_payload


class InspectUniGroundV2PayloadTest(unittest.TestCase):
    def test_inspect_payload_passes_on_well_formed_balanced_payload(self):
        payload = {
            "image_embeddings": torch.nn.functional.normalize(torch.randn(6, 8), dim=-1),
            "candidate_embeddings": torch.nn.functional.normalize(torch.randn(6, 8), dim=-1),
            "prefix_embeddings": torch.nn.functional.normalize(torch.randn(6, 8), dim=-1),
            "region_embeddings": torch.nn.functional.normalize(torch.randn(6, 2, 8), dim=-1),
            "labels": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "metadata": {
                "record_count": 6,
                "image_count": 2,
                "candidate_vocab_size": 6,
                "prefix_count": 6,
                "augmentation_policy": {"llm_used": False},
                "region_mode": "gt_bbox_topr",
                "label_counts": {"support": 2, "contradiction": 2, "abstain": 2},
            },
        }
        report = inspect_payload(payload)
        self.assertTrue(report["passed"])
        self.assertEqual(report["label_counts"]["support"], 2)
        self.assertEqual(report["region_shape"], [6, 2, 8])


if __name__ == "__main__":
    unittest.main()
