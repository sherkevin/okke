from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import (
    UniversalCandidate,
    UniversalObservation,
    UniversalPluginConfig,
    UniversalPluginOutput,
)
from uniground_v2.runtime import RetrievalResult, TriggerDecision, UniGroundV2LogitsProcessor


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "prefix"


class _AlwaysTrigger:
    def decide(self, scores, input_ids, tokenizer):
        return TriggerDecision(fire=True, reason="entropy_high")


class _CandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text):
        ids = top_ids.tolist()
        return [
            UniversalCandidate(ids[0], "tok1", "dog", "dog", True),
            UniversalCandidate(ids[1], "tok2", "cat", "cat", True),
        ]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class _Encoder:
    def encode_texts(self, texts):
        rows = []
        for idx, _text in enumerate(texts):
            rows.append(torch.tensor([float(idx + 1), 0.0, 0.0, 0.0]))
        return torch.stack(rows, dim=0)


class _Retriever:
    def retrieve(self, observation, candidates, device):
        return RetrievalResult(observation=UniversalObservation(image_embedding=torch.ones(4, device=device)), metadata={"mode": "identity"})


class _Scorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        return UniversalPluginOutput(
            support=torch.tensor([3.0, -3.0], device=candidate_embeddings.device),
            contradiction=torch.tensor([-3.0, 3.0], device=candidate_embeddings.device),
            abstain=torch.tensor([-8.0, -8.0], device=candidate_embeddings.device),
        )


class RuntimeBoundedScatterTest(unittest.TestCase):
    def test_only_topk_positions_change(self):
        cfg = UniversalPluginConfig(top_k=2, bias_scale=0.5)
        processor = UniGroundV2LogitsProcessor(
            config=cfg,
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=UniversalObservation(image_embedding=torch.zeros(4)),
            trigger=_AlwaysTrigger(),
            candidate_builder=_CandidateBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, 0.9, 0.8, 0.2, 0.05]], dtype=torch.float32)
        updated = processor(input_ids, scores)

        changed_positions = [idx for idx in range(scores.shape[-1]) if not torch.isclose(updated[0, idx], scores[0, idx])]
        self.assertEqual(changed_positions, [1, 2])
        self.assertNotEqual(float(updated[0, 1]), float(scores[0, 1]))
        self.assertNotEqual(float(updated[0, 2]), float(scores[0, 2]))


if __name__ == "__main__":
    unittest.main()
