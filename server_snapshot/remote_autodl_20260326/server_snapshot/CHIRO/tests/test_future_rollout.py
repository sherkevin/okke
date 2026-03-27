import unittest

import torch

from chord.future_rollout import (
    FutureRolloutResult,
    greedy_future_rollout,
    greedy_future_rollout_from_bootstrap,
    safe_rollout_future,
)


class FutureRolloutTests(unittest.TestCase):
    def test_greedy_future_rollout_accumulates_visual_and_text_sums(self) -> None:
        fixtures = {
            6: (
                42,
                torch.tensor([0.1, 0.1, 0.2, 0.3, 0.1, 0.2], dtype=torch.float32),
            ),
            7: (
                99,
                torch.tensor([0.05, 0.05, 0.1, 0.2, 0.1, 0.15, 0.35], dtype=torch.float32),
            ),
        }

        def _step_fn(prefix_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
            return fixtures[int(prefix_ids.shape[0])]

        trace = greedy_future_rollout(
            prefix_ids=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
            step_fn=_step_fn,
            visual_token_indices=torch.tensor([2, 3], dtype=torch.long),
            image_token_span=(2, 3),
            token_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
            horizon=2,
            lambda_txt=1.0,
            eps=1e-6,
        )

        self.assertEqual(trace.generated_tokens, (42, 99))
        self.assertAlmostEqual(trace.step_visual_sums[0], 0.8)
        self.assertAlmostEqual(trace.step_visual_sums[1], 0.5)
        self.assertAlmostEqual(trace.step_text_sums[0], 0.5)
        self.assertAlmostEqual(trace.step_text_sums[1], 0.7)
        self.assertAlmostEqual(trace.result.sum_visual, 1.3)
        self.assertAlmostEqual(trace.result.sum_text, 1.2)
        self.assertAlmostEqual(trace.result.f_future, 0.1)
        self.assertAlmostEqual(trace.result.r_future, 1.3 / (1.3 + 1.2 + 1e-6))

    def test_safe_rollout_future_returns_zero_result_on_structural_failure(self) -> None:
        def _broken_rollout() -> FutureRolloutResult:
            raise ValueError("broken rollout")

        result = safe_rollout_future(_broken_rollout)

        self.assertTrue(result.failed)
        self.assertAlmostEqual(result.sum_visual, 0.0)
        self.assertAlmostEqual(result.sum_text, 0.0)
        self.assertAlmostEqual(result.f_future, 0.0)
        self.assertAlmostEqual(result.r_future, 0.0)

    def test_bootstrap_rollout_reuses_first_step_and_continues_greedily(self) -> None:
        fixtures = {
            7: (
                99,
                torch.tensor([0.05, 0.05, 0.1, 0.2, 0.1, 0.15, 0.35], dtype=torch.float32),
            ),
        }

        def _continuation_step_fn(prefix_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
            return fixtures[int(prefix_ids.shape[0])]

        trace = greedy_future_rollout_from_bootstrap(
            prefix_ids=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
            bootstrap_next_token_id=42,
            bootstrap_attn_last_row=torch.tensor([0.1, 0.1, 0.2, 0.3, 0.1, 0.2], dtype=torch.float32),
            continuation_step_fn=_continuation_step_fn,
            visual_token_indices=torch.tensor([2, 3], dtype=torch.long),
            image_token_span=(2, 3),
            token_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
            horizon=2,
            lambda_txt=1.0,
            eps=1e-6,
        )

        self.assertEqual(trace.generated_tokens, (42, 99))
        self.assertAlmostEqual(trace.step_visual_sums[0], 0.8)
        self.assertAlmostEqual(trace.step_visual_sums[1], 0.5)
        self.assertAlmostEqual(trace.step_text_sums[0], 0.5)
        self.assertAlmostEqual(trace.step_text_sums[1], 0.7)
        self.assertAlmostEqual(trace.result.sum_visual, 1.3)
        self.assertAlmostEqual(trace.result.sum_text, 1.2)
        self.assertAlmostEqual(trace.result.f_future, 0.1)

    def test_bootstrap_rollout_preserves_multimodal_visual_indices_when_attention_is_longer(self) -> None:
        trace = greedy_future_rollout_from_bootstrap(
            prefix_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
            bootstrap_next_token_id=42,
            bootstrap_attn_last_row=torch.tensor(
                [0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.05, 0.05],
                dtype=torch.float32,
            ),
            continuation_step_fn=lambda prefix_ids: (99, torch.ones(prefix_ids.shape[0], dtype=torch.float32)),
            visual_token_indices=torch.tensor([4, 5], dtype=torch.long),
            image_token_span=(4, 5),
            token_weights=torch.tensor([1.0, 2.0], dtype=torch.float32),
            horizon=1,
            lambda_txt=1.0,
            eps=1e-6,
        )

        self.assertEqual(trace.generated_tokens, (42,))
        self.assertAlmostEqual(trace.step_visual_sums[0], 1.1)
        self.assertAlmostEqual(trace.step_text_sums[0], 0.6)
        self.assertAlmostEqual(trace.result.sum_visual, 1.1)
        self.assertAlmostEqual(trace.result.sum_text, 0.6)
        self.assertAlmostEqual(trace.result.f_future, 0.5)


if __name__ == "__main__":
    unittest.main()
