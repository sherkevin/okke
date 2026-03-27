from __future__ import annotations

import unittest
import torch

from bra_universal_plugin import StringStructuralGate
from uniground_v2.bridge import CandidateFrontierBuilder


class _Tokenizer:
    vocab = {
        1: "ju",
        2: "##mp",
        3: " dog",
    }

    def convert_ids_to_tokens(self, token_id):
        return self.vocab[int(token_id)]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.vocab[int(token_id)] for token_id in ids)


class BridgePrefixStateCompletionTest(unittest.TestCase):
    def test_prefix_state_completes_continuation_candidate(self):
        builder = CandidateFrontierBuilder()
        tokenizer = _Tokenizer()
        gate = StringStructuralGate()

        builder.sync_prefix_state(tokenizer, [[1]])
        candidates = builder.build(tokenizer, torch.tensor([2, 3]), gate, prefix_text="ju")

        jump_candidate = candidates[0]
        self.assertEqual(jump_candidate.metadata["status"], "complete")
        self.assertEqual(jump_candidate.span_text, "jump")
        self.assertTrue(jump_candidate.is_groundable)


if __name__ == "__main__":
    unittest.main()
