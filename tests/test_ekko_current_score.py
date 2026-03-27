from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKKO"))

from chord.chord_fusion import apply_current_chord_score  # noqa: E402


class EkkoCurrentScoreTests(unittest.TestCase):
    def test_current_score_adds_weighted_visual_support(self) -> None:
        updated_scores, v_anchor = apply_current_chord_score(
            opera_scores=torch.tensor([0.3, 0.1], dtype=torch.float32),
            candidate_visual_attn=torch.tensor(
                [
                    [0.5, 0.2, 0.1, 0.2],
                    [0.1, 0.4, 0.3, 0.2],
                ],
                dtype=torch.float32,
            ),
            token_weights=torch.tensor([1.4, 1.0, 1.0, 1.0], dtype=torch.float32),
            lambda_cur=0.25,
        )

        self.assertTrue(torch.allclose(v_anchor, torch.tensor([0.2, 0.04], dtype=torch.float32)))
        self.assertTrue(torch.allclose(updated_scores, torch.tensor([0.35, 0.11], dtype=torch.float32)))

    def test_zero_lambda_preserves_opera_scores_exactly(self) -> None:
        opera_scores = torch.tensor([0.8, 0.2], dtype=torch.float32)
        updated_scores, v_anchor = apply_current_chord_score(
            opera_scores=opera_scores,
            candidate_visual_attn=torch.tensor(
                [
                    [0.5, 0.2, 0.1, 0.2],
                    [0.1, 0.4, 0.3, 0.2],
                ],
                dtype=torch.float32,
            ),
            token_weights=torch.tensor([1.4, 1.0, 1.0, 1.0], dtype=torch.float32),
            lambda_cur=0.0,
        )

        self.assertTrue(torch.equal(updated_scores, opera_scores))
        self.assertTrue(torch.equal(v_anchor, torch.zeros_like(opera_scores)))

    def test_uniform_weights_produce_zero_kernel_bonus(self) -> None:
        updated_scores, v_anchor = apply_current_chord_score(
            opera_scores=torch.tensor([0.3, 0.1], dtype=torch.float32),
            candidate_visual_attn=torch.tensor(
                [
                    [0.5, 0.2, 0.1, 0.2],
                    [0.1, 0.4, 0.3, 0.2],
                ],
                dtype=torch.float32,
            ),
            token_weights=torch.ones(4, dtype=torch.float32),
            lambda_cur=0.25,
        )

        self.assertTrue(torch.equal(v_anchor, torch.zeros(2, dtype=torch.float32)))
        self.assertTrue(torch.allclose(updated_scores, torch.tensor([0.3, 0.1], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
