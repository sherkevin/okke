from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import UniversalCandidate, UniversalObservation
from uniground_v2.regions import RegionRetriever


class _Encoder:
    def encode_texts(self, texts):
        mapping = {
            "dog": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "cat": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "a photo of dog": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
        return torch.stack([mapping[text] for text in texts], dim=0)


class RegionRetrievalTopRTest(unittest.TestCase):
    def test_retriever_selects_top_regions_per_candidate(self):
        retriever = RegionRetriever(_Encoder(), top_r=1)
        observation = UniversalObservation(
            image_embedding=torch.tensor([0.5, 0.5], dtype=torch.float32),
            region_embeddings=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            metadata={"region_mode": "detector_regions"},
        )
        candidates = [
            UniversalCandidate(1, "tok1", "dog", "dog", True),
            UniversalCandidate(2, "tok2", "cat", "cat", True),
        ]
        result = retriever.retrieve(observation, candidates, torch.device("cpu"))
        self.assertEqual(result.metadata["selected_region_indices_per_candidate"], [[0], [2]])
        self.assertEqual(tuple(result.observation.region_embeddings.shape), (2, 1, 2))

    def test_retriever_can_use_task_query_scope_without_pope_specific_mode(self):
        retriever = RegionRetriever(_Encoder(), top_r=1)
        observation = UniversalObservation(
            image_embedding=torch.tensor([0.5, 0.5], dtype=torch.float32),
            region_embeddings=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            metadata={
                "region_mode": "detector_regions",
                "runtime_context": {
                    "task_name": "custom_binary",
                    "retrieval_scope": "task_query",
                    "retrieval_query_text": "a photo of dog",
                },
            },
        )
        candidates = [
            UniversalCandidate(1, "tok1", "irrelevant one", "irrelevant one", True),
            UniversalCandidate(2, "tok2", "irrelevant two", "irrelevant two", True),
        ]
        result = retriever.retrieve(observation, candidates, torch.device("cpu"))
        self.assertEqual(result.metadata["selected_region_indices_per_candidate"], [[0], [0]])


if __name__ == "__main__":
    unittest.main()
