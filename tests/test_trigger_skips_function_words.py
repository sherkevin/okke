from __future__ import annotations

import unittest

import torch

from uniground_v2.trigger import EntropyMarginTrigger


class _Tokenizer:
    MAP = {
        0: "the",
        1: "and",
        2: "of",
        3: "to",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            return self.MAP[token_ids]
        if isinstance(token_ids, list):
            return "".join(self.MAP[int(token_id)] for token_id in token_ids)
        return self.MAP[int(token_ids)]


class TriggerFunctionWordGateTest(unittest.TestCase):
    def test_function_word_frontier_does_not_fire(self):
        trigger = EntropyMarginTrigger(top_k=4, entropy_threshold=0.1, margin_threshold=1.0, min_content_ratio=0.25)
        scores = torch.tensor([[1.0, 0.99, 0.98, 0.97]], dtype=torch.float32)
        decision = trigger.decide(scores, torch.tensor([[1, 2]]), _Tokenizer())
        self.assertFalse(decision.fire)
        self.assertEqual(decision.reason, "content_low")


if __name__ == "__main__":
    unittest.main()
