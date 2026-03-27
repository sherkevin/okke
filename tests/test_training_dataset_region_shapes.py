from __future__ import annotations

import unittest

import torch

from uniground_v2.training_dataset import PsiUnivV2Dataset, V2TrainingTensors


class TrainingDatasetRegionShapesTest(unittest.TestCase):
    def test_dataset_returns_region_tensor_per_example(self):
        tensors = V2TrainingTensors(
            image_embeddings=torch.randn(3, 8),
            hypothesis_embeddings=torch.randn(3, 8),
            query_embeddings=torch.randn(3, 8),
            region_embeddings=torch.randn(3, 2, 8),
            labels=torch.randint(0, 3, (3,)),
            metadata={},
        )
        dataset = PsiUnivV2Dataset(tensors)
        item = dataset[1]
        self.assertEqual(tuple(item["region_embeddings"].shape), (2, 8))
        self.assertEqual(tuple(item["image_embeddings"].shape), (8,))
        self.assertEqual(tuple(item["hypothesis_embeddings"].shape), (8,))
        self.assertEqual(tuple(item["query_embeddings"].shape), (8,))


if __name__ == "__main__":
    unittest.main()
