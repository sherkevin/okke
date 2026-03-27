from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import UniversalObservation
from uniground_v2.scorer import HardcodedUniversalScorer


class HardcodedScorerTest(unittest.TestCase):
    def test_hardcoded_scorer_returns_vector_outputs(self):
        scorer = HardcodedUniversalScorer()
        observation = UniversalObservation(
            image_embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            region_embeddings=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )
        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        prefix = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        output = scorer(observation, candidates, prefix)
        self.assertEqual(tuple(output.support.shape), (2,))
        self.assertEqual(tuple(output.contradiction.shape), (2,))
        self.assertEqual(tuple(output.abstain.shape), (2,))
        self.assertGreater(float(output.support[0]), float(output.support[1]))


if __name__ == "__main__":
    unittest.main()
