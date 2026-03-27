from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import UniversalObservation, UniversalPluginConfig
from uniground_v2.runtime import TriggerDecision, UniGroundV2LogitsProcessor


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "prefix"


class _NeverTrigger:
    def decide(self, scores, input_ids, tokenizer):
        return TriggerDecision(fire=False, reason="entropy_low", entropy=0.1, margin=0.9)


class _Encoder:
    def encode_texts(self, texts):
        raise AssertionError("encoder should not be used when trigger is off")


class _Scorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        raise AssertionError("scorer should not be used when trigger is off")


class RuntimeNoOpTest(unittest.TestCase):
    def test_returns_original_scores_when_trigger_does_not_fire(self):
        cfg = UniversalPluginConfig(top_k=3)
        processor = UniGroundV2LogitsProcessor(
            config=cfg,
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=UniversalObservation(image_embedding=torch.zeros(4)),
            trigger=_NeverTrigger(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        updated = processor(input_ids, scores)
        self.assertTrue(torch.equal(updated, scores))
        summary = processor.get_summary_stats()
        self.assertEqual(summary["intervention_coverage"], 0.0)
        self.assertEqual(summary["trigger_fire_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
