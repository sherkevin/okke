from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import StringStructuralGate
from uniground_v2.bridge import CandidateFrontierBuilder


class _Tokenizer:
    MAP = {
        1: "A",
        2: "a",
        3: "B",
        4: "yes",
        5: "no",
        6: "Ġyes",
        7: "Ġno",
    }

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, list):
            return [self.MAP[int(token_id)] for token_id in token_ids]
        return self.MAP[int(token_ids)]


class BridgeTaskAdapterBinaryTest(unittest.TestCase):
    def test_binary_adapter_maps_uppercase_options_only(self):
        builder = CandidateFrontierBuilder()
        candidates = builder.build(
            _Tokenizer(),
            torch.tensor([1, 2, 3]),
            StringStructuralGate(),
            prefix_text="Is there a dog in the image?\nOptions:\nA. yes\nB. no\nAnswer with the option letter only.",
        )
        self.assertEqual(candidates[0].span_text, "a photo containing dog")
        self.assertEqual(candidates[2].span_text, "a photo without dog")
        self.assertEqual(candidates[1].normalized_text, "a")
        self.assertIsNone(candidates[1].metadata["semantic_alias"])

    def test_pope_yes_no_tokens_map_to_object_hypotheses(self):
        builder = CandidateFrontierBuilder()
        candidates = builder.build(
            _Tokenizer(),
            torch.tensor([4, 5, 6, 7]),
            StringStructuralGate(),
            prefix_text="Is there a snowboard in the image?\nAnswer with exactly one word: yes or no.",
        )
        self.assertEqual(candidates[0].span_text, "a photo containing snowboard")
        self.assertEqual(candidates[1].span_text, "a photo without snowboard")
        self.assertEqual(candidates[2].span_text, "a photo containing snowboard")
        self.assertEqual(candidates[3].span_text, "a photo without snowboard")
        self.assertTrue(candidates[0].is_groundable)
        self.assertTrue(candidates[1].is_groundable)

    def test_context_driven_verifier_mode_keeps_only_binary_targets_groundable(self):
        builder = CandidateFrontierBuilder()
        candidates = builder.build(
            _Tokenizer(),
            torch.tensor([2, 6, 7]),
            StringStructuralGate(),
            prefix_text="prefix without explicit object question",
            context={
                "task_name": "pope",
                "decision_mode": "answer_labels",
                "answer_mode": "yes_no",
                "object_label": "snowboard",
                "hypothesis_text_by_label": {
                    "yes": "a photo containing snowboard",
                    "no": "a photo without snowboard",
                },
            },
        )
        self.assertFalse(candidates[0].is_groundable)
        self.assertEqual(candidates[1].span_text, "a photo containing snowboard")
        self.assertEqual(candidates[2].span_text, "a photo without snowboard")
        self.assertTrue(candidates[1].is_groundable)
        self.assertTrue(candidates[2].is_groundable)


if __name__ == "__main__":
    unittest.main()
