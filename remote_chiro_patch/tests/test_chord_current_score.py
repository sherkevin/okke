import unittest

import torch

from chord.chord_fusion import apply_current_chord_score


class CHORDCurrentScoreTests(unittest.TestCase):
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

        self.assertTrue(torch.allclose(v_anchor, torch.tensor([1.2, 1.04], dtype=torch.float32)))
        self.assertTrue(torch.allclose(updated_scores, torch.tensor([0.6, 0.36], dtype=torch.float32)))

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


if __name__ == "__main__":
    unittest.main()
