from __future__ import annotations

import unittest

import torch

from uniground_v2.trigger import EntropyMarginTrigger


class _Tokenizer:
    MAP = {
        0: "dog",
        1: "cat",
        2: "horse",
        3: "tree",
        4: "yes",
        5: "no",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            return self.MAP[token_ids]
        if isinstance(token_ids, list):
            return "".join(self.MAP[int(token_id)] for token_id in token_ids)
        return self.MAP[int(token_ids)]


class TriggerEntropyGateTest(unittest.TestCase):
    def test_low_entropy_does_not_fire(self):
        trigger = EntropyMarginTrigger(top_k=4, entropy_threshold=0.8, margin_threshold=10.0, min_content_ratio=0.1)
        scores = torch.tensor([[8.0, -4.0, -5.0, -6.0]], dtype=torch.float32)
        decision = trigger.decide(scores, torch.tensor([[1, 2]]), _Tokenizer())
        self.assertFalse(decision.fire)
        self.assertEqual(decision.reason, "entropy_low")

    def test_high_entropy_can_fire(self):
        trigger = EntropyMarginTrigger(top_k=4, entropy_threshold=1.0, margin_threshold=0.3, min_content_ratio=0.1)
        scores = torch.tensor([[1.0, 0.95, 0.9, 0.85]], dtype=torch.float32)
        decision = trigger.decide(scores, torch.tensor([[1, 2]]), _Tokenizer())
        self.assertTrue(decision.fire)
        self.assertEqual(decision.reason, "entropy_margin_fire")

    def test_binary_object_query_can_fire_even_when_entropy_is_low(self):
        class _BinaryTokenizer(_Tokenizer):
            def decode(self, token_ids, skip_special_tokens=False):
                if isinstance(token_ids, int):
                    return self.MAP[token_ids]
                return "Is there a dog in the image?\nAnswer with exactly one word: yes or no."

        trigger = EntropyMarginTrigger(top_k=4, entropy_threshold=10.0, margin_threshold=0.01, min_content_ratio=0.5)
        scores = torch.tensor([[8.0, -4.0, -5.0, -6.0, 7.8, 6.9]], dtype=torch.float32)
        decision = trigger.decide(scores, torch.tensor([[1, 2, 3]]), _BinaryTokenizer())
        self.assertTrue(decision.fire)
        self.assertEqual(decision.reason, "binary_object_query")


if __name__ == "__main__":
    unittest.main()
