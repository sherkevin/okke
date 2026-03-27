from __future__ import annotations

import unittest

import torch

from uniground_v2.training_dataset import load_training_tensors_v2


class TrainingPayloadSchemaV2Test(unittest.TestCase):
    def test_payload_with_query_hypothesis_embeddings_is_accepted(self):
        payload = {
            "image_embeddings": torch.randn(4, 8),
            "hypothesis_embeddings": torch.randn(4, 8),
            "query_embeddings": torch.randn(4, 8),
            "region_embeddings": torch.randn(4, 2, 8),
            "labels": torch.randint(0, 3, (4,)),
            "metadata": {"record_count": 4},
        }
        tensors = load_training_tensors_v2(payload)
        self.assertEqual(tuple(tensors.region_embeddings.shape), (4, 2, 8))
        self.assertEqual(tuple(tensors.hypothesis_embeddings.shape), (4, 8))
        self.assertEqual(tuple(tensors.query_embeddings.shape), (4, 8))
        self.assertEqual(tensors.metadata["record_count"], 4)

    def test_payload_with_region_embeddings_is_accepted(self):
        payload = {
            "image_embeddings": torch.randn(4, 8),
            "candidate_embeddings": torch.randn(4, 8),
            "prefix_embeddings": torch.randn(4, 8),
            "region_embeddings": torch.randn(4, 2, 8),
            "labels": torch.randint(0, 3, (4,)),
            "metadata": {"record_count": 4},
        }
        tensors = load_training_tensors_v2(payload)
        self.assertEqual(tuple(tensors.region_embeddings.shape), (4, 2, 8))
        self.assertEqual(tuple(tensors.hypothesis_embeddings.shape), (4, 8))
        self.assertEqual(tuple(tensors.query_embeddings.shape), (4, 8))
        self.assertEqual(tensors.metadata["record_count"], 4)


if __name__ == "__main__":
    unittest.main()
