from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import UniversalCandidate, UniversalObservation, UniversalPluginConfig, UniversalPluginOutput
from uniground_v2.runtime import RetrievalResult, TriggerDecision, UniGroundV2LogitsProcessor


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "prefix"


class _AlwaysTrigger:
    def decide(self, scores, input_ids, tokenizer):
        return TriggerDecision(fire=True, reason="entropy_margin_fire", entropy=2.0, margin=0.1, content_ratio=1.0)


class _CandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text):
        return [UniversalCandidate(int(top_ids[0]), "tok", "dog", "dog", True)]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class _Encoder:
    def encode_texts(self, texts):
        return torch.ones((len(texts), 4), dtype=torch.float32)


class _Retriever:
    def retrieve(self, observation, candidates, device):
        return RetrievalResult(observation=UniversalObservation(image_embedding=torch.ones(4, device=device)), metadata={"region_mode": "none"})


class _HighAbstainScorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        return UniversalPluginOutput(
            support=torch.tensor([1.0], device=candidate_embeddings.device),
            contradiction=torch.tensor([0.0], device=candidate_embeddings.device),
            abstain=torch.tensor([10.0], device=candidate_embeddings.device),
        )


class RuntimeAbstainBackoffTest(unittest.TestCase):
    def test_high_abstain_returns_original_scores_and_records_backoff(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=1, abstain_threshold=0.55),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_HighAbstainScorer(),
            observation=UniversalObservation(image_embedding=torch.zeros(4)),
            trigger=_AlwaysTrigger(),
            candidate_builder=_CandidateBuilder(),
            retriever=_Retriever(),
        )
        scores = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        updated = processor(torch.tensor([[1, 2, 3]]), scores)
        self.assertTrue(torch.equal(updated, scores))
        summary = processor.get_summary_stats()
        self.assertEqual(summary["abort_trigger_rate"], 1.0)
        self.assertEqual(summary["abort_backoff_verified_steps"], 1)


if __name__ == "__main__":
    unittest.main()
