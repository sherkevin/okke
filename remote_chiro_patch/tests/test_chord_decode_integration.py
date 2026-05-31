import unittest

import torch

from chord.chord_fusion import apply_chord_rerank, fuse_chord_scores
from chord.future_rollout import FutureRolloutResult, safe_rollout_future


class CHORDDecodeIntegrationTests(unittest.TestCase):
    def test_full_chord_fusion_matches_spec_formula(self) -> None:
        fused = fuse_chord_scores(
            opera_scores=torch.tensor([0.2, 0.1], dtype=torch.float32),
            v_anchor=torch.tensor([0.8, 0.0], dtype=torch.float32),
            f_future=torch.tensor([0.0, 0.6], dtype=torch.float32),
            lambda_cur=0.25,
            lambda_fut=0.5,
        )

        expected = torch.tensor([0.4, 0.4], dtype=torch.float32)
        self.assertTrue(torch.allclose(fused, expected))

    def test_decode_rerank_uses_fused_scores_when_no_rollback_occurs(self) -> None:
        result = apply_chord_rerank(
            candidate_tokens=torch.tensor([10, 20, 30], dtype=torch.long),
            opera_scores=torch.tensor([0.3, 0.29, 0.28], dtype=torch.float32),
            v_anchor=torch.tensor([0.0, 0.4, 0.0], dtype=torch.float32),
            f_future=torch.tensor([0.0, 0.0, 0.4], dtype=torch.float32),
            lambda_cur=0.25,
            lambda_fut=0.5,
            rollback_triggered=False,
        )

        self.assertFalse(result.rollback_triggered)
        self.assertTrue(result.used_chord)
        self.assertTrue(torch.equal(result.ranked_candidate_tokens, torch.tensor([30, 20, 10], dtype=torch.long)))

    def test_decode_rerank_preserves_opera_order_when_weights_are_zero(self) -> None:
        result = apply_chord_rerank(
            candidate_tokens=torch.tensor([10, 20, 30], dtype=torch.long),
            opera_scores=torch.tensor([0.31, 0.3, 0.29], dtype=torch.float32),
            v_anchor=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            f_future=torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32),
            lambda_cur=0.0,
            lambda_fut=0.0,
            rollback_triggered=False,
        )

        self.assertFalse(result.rollback_triggered)
        self.assertFalse(result.recompute_after_rollback)
        self.assertTrue(torch.equal(result.ranked_candidate_tokens, torch.tensor([10, 20, 30], dtype=torch.long)))

    def test_decode_rerank_preserves_opera_candidates_when_rollback_triggers(self) -> None:
        result = apply_chord_rerank(
            candidate_tokens=torch.tensor([10, 20, 30], dtype=torch.long),
            opera_scores=torch.tensor([0.31, 0.3, 0.29], dtype=torch.float32),
            v_anchor=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            f_future=torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32),
            lambda_cur=0.25,
            lambda_fut=0.5,
            rollback_triggered=True,
        )

        self.assertTrue(result.rollback_triggered)
        self.assertFalse(result.used_chord)
        self.assertTrue(result.recompute_after_rollback)
        self.assertTrue(torch.equal(result.ranked_candidate_tokens, torch.tensor([10, 20, 30], dtype=torch.long)))

    def test_full_fallback_returns_exact_opera_scores(self) -> None:
        opera_scores = torch.tensor([0.31, 0.3, 0.29], dtype=torch.float32)
        fused = fuse_chord_scores(
            opera_scores=opera_scores,
            v_anchor=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            f_future=torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32),
            lambda_cur=0.0,
            lambda_fut=0.0,
        )

        self.assertTrue(torch.equal(fused, opera_scores))
        self.assertIsNot(fused, opera_scores)

    def test_future_failure_is_reported_non_silently(self) -> None:
        def failing_rollout() -> FutureRolloutResult:
            raise RuntimeError("synthetic future OOM")

        result = safe_rollout_future(failing_rollout)

        self.assertIsInstance(result, FutureRolloutResult)
        self.assertTrue(result.failed)
        self.assertEqual(result.f_future, 0.0)
        self.assertIn("synthetic future OOM", result.error_message)


if __name__ == "__main__":
    unittest.main()
