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
        return TriggerDecision(fire=True, reason="entropy_high", entropy=2.0, margin=0.1, content_ratio=1.0)


class _CandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text):
        ids = top_ids.tolist()
        return [UniversalCandidate(ids[0], "tok1", "dog", "dog", True)]

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


class _Scorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        return UniversalPluginOutput(
            support=torch.tensor([1.0], device=candidate_embeddings.device),
            contradiction=torch.tensor([0.0], device=candidate_embeddings.device),
            abstain=torch.tensor([-8.0], device=candidate_embeddings.device),
        )


class RuntimeAuditSchemaTest(unittest.TestCase):
    def test_audit_and_summary_schema(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=1),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=UniversalObservation(image_embedding=torch.zeros(4)),
            trigger=_AlwaysTrigger(),
            candidate_builder=_CandidateBuilder(),
            retriever=_Retriever(),
        )
        updated = processor(torch.tensor([[1, 2, 3]]), torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
        self.assertEqual(updated.shape[-1], 3)

        audit = processor.get_audit_log()
        self.assertEqual(len(audit), 1)
        entry = audit[0]
        self.assertIn("prefix_text", entry)
        self.assertIn("trigger", entry)
        self.assertIn("candidates", entry)
        self.assertIn("retrieval", entry)
        self.assertIn("bias", entry)
        self.assertIn("intervened", entry)

        summary = processor.get_summary_stats()
        self.assertIn("intervention_coverage", summary)
        self.assertIn("trigger_fire_rate", summary)
        self.assertIn("avg_active_candidates", summary)
        self.assertIn("latency_split", summary)
        self.assertIn("candidate_construction_ms", summary["latency_split"])


if __name__ == "__main__":
    unittest.main()
