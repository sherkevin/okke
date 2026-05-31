from __future__ import annotations

import torch

from uniground_v2.trigger import EntropyMarginTrigger


class _Tokenizer:
    MAP = {
        1: "prefix",
        2: "Ġyes",
        3: "Ġno",
        4: "dog",
    }

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, list):
            return [self.MAP[int(token_id)] for token_id in token_ids]
        return self.MAP[int(token_ids)]

    def decode(self, ids, skip_special_tokens=True):
        return "Is there a dog in the image?\nAnswer with exactly one word: yes or no."


def test_verifier_trigger_accepts_prefixed_yes_no_tokens():
    trigger = EntropyMarginTrigger(top_k=2)
    scores = torch.tensor([[0.0, 5.0, 4.5, -1.0]], dtype=torch.float32)
    input_ids = torch.tensor([[1, 1, 1]], dtype=torch.long)
    decision = trigger.decide(
        scores,
        input_ids,
        _Tokenizer(),
        context={"decision_mode": "answer_labels", "prompt_token_count": 3},
    )
    assert decision.fire is True
    assert decision.reason == "answer_label_step"


def test_verifier_trigger_stops_after_first_generated_token():
    trigger = EntropyMarginTrigger(top_k=2)
    scores = torch.tensor([[0.0, 5.0, 4.5, -1.0]], dtype=torch.float32)
    input_ids = torch.tensor([[1, 1, 1, 2]], dtype=torch.long)
    decision = trigger.decide(
        scores,
        input_ids,
        _Tokenizer(),
        context={"decision_mode": "answer_labels", "prompt_token_count": 3},
    )
    assert decision.fire is False
    assert decision.reason == "after_answer_step"
