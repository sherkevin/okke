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
    }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            return self.MAP[token_ids]
        if isinstance(token_ids, list):
            return "".join(self.MAP[int(token_id)] for token_id in token_ids)
        return self.MAP[int(token_ids)]


class TriggerMarginGateTest(unittest.TestCase):
    def test_high_margin_blocks_even_if_entropy_is_high_enough(self):
        trigger = EntropyMarginTrigger(top_k=4, entropy_threshold=0.2, margin_threshold=0.5, min_content_ratio=0.1)
        scores = torch.tensor([[2.0, 1.0, 0.99, 0.98]], dtype=torch.float32)
        decision = trigger.decide(scores, torch.tensor([[1, 2]]), _Tokenizer())
        self.assertFalse(decision.fire)
        self.assertEqual(decision.reason, "margin_high")


if __name__ == "__main__":
    unittest.main()
