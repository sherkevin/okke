from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKKO"))

from chord.anchor_builder import build_visual_token_weights  # noqa: E402
from chord.chord_fusion import apply_chord_rerank, fuse_chord_scores  # noqa: E402
from chord.future_rollout import FutureRolloutResult, compute_future_trajectory_score, safe_rollout_future  # noqa: E402


class EkkoTheoryLockTests(unittest.TestCase):
    def test_lambda_zero_falls_back_exactly_to_opera_scores(self) -> None:
        opera_scores = torch.tensor([1.25, -0.5, 0.75], dtype=torch.float32)
        v_anchor = torch.tensor([0.1, 0.9, 0.5], dtype=torch.float32)
        f_future = torch.tensor([0.6, -0.2, 0.3], dtype=torch.float32)

        fused = fuse_chord_scores(
            opera_scores=opera_scores,
            v_anchor=v_anchor,
            f_future=f_future,
            lambda_cur=0.0,
            lambda_fut=0.0,
        )

        self.assertTrue(torch.equal(fused, opera_scores))

    def test_zero_anchor_penalty_only_hits_unsupported_candidates(self) -> None:
        fused = fuse_chord_scores(
            opera_scores=torch.tensor([0.3, 0.3], dtype=torch.float32),
            v_anchor=torch.tensor([0.0, 0.2], dtype=torch.float32),
            f_future=torch.tensor([0.0, 0.0], dtype=torch.float32),
            lambda_cur=0.0,
            lambda_fut=0.0,
            zero_anchor_penalty=0.5,
        )

        self.assertTrue(torch.allclose(fused, torch.tensor([-0.2, 0.3], dtype=torch.float32)))

    def test_detector_failure_returns_uniform_visual_weights(self) -> None:
        weights = build_visual_token_weights(
            membership=torch.zeros((0, 4), dtype=torch.float32),
            relevance=torch.zeros(0, dtype=torch.float32),
            confidence=torch.zeros(0, dtype=torch.float32),
            alpha_anchor=0.5,
        )

        self.assertTrue(torch.equal(weights, torch.ones(4, dtype=torch.float32)))

    def test_future_score_uses_sums_and_counts_generated_text_tokens(self) -> None:
        result = compute_future_trajectory_score(
            visual_attentions=[
                torch.tensor([0.2, 0.3], dtype=torch.float32),
                torch.tensor([0.1, 0.5], dtype=torch.float32),
            ],
            text_attentions=[
                torch.tensor([0.1, 0.05], dtype=torch.float32),
                torch.tensor([0.2, 0.15, 0.25], dtype=torch.float32),
            ],
            token_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
            lambda_txt=1.0,
            eps=1e-6,
        )

        self.assertAlmostEqual(result.sum_visual, 1.9)
        self.assertAlmostEqual(result.sum_text, 0.75)
        self.assertAlmostEqual(result.f_future, 1.15)
        self.assertAlmostEqual(result.r_future, 1.9 / (1.9 + 0.75 + 1e-6))

    def test_future_rollout_failure_falls_back_to_zero_without_throwing(self) -> None:
        def _broken_rollout() -> FutureRolloutResult:
            raise RuntimeError("synthetic rollout failure")

        result = safe_rollout_future(_broken_rollout)

        self.assertTrue(result.failed)
        self.assertAlmostEqual(result.sum_visual, 0.0)
        self.assertAlmostEqual(result.sum_text, 0.0)
        self.assertAlmostEqual(result.f_future, 0.0)
        self.assertAlmostEqual(result.r_future, 0.0)

    def test_rollback_precedence_discards_precomputed_chord_scores(self) -> None:
        rerank_result = apply_chord_rerank(
            candidate_tokens=torch.tensor([11, 22, 33], dtype=torch.long),
            opera_scores=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            v_anchor=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            f_future=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
            lambda_cur=0.25,
            lambda_fut=0.5,
            zero_anchor_penalty=0.0,
            rollback_triggered=True,
        )

        self.assertTrue(rerank_result.rollback_triggered)
        self.assertFalse(rerank_result.used_chord)
        self.assertTrue(rerank_result.recompute_after_rollback)
        self.assertTrue(torch.equal(rerank_result.ranked_candidate_tokens, torch.tensor([11, 22, 33], dtype=torch.long)))
        self.assertIsNone(rerank_result.fused_scores)


if __name__ == "__main__":
    unittest.main()
