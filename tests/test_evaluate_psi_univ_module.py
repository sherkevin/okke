from __future__ import annotations

import unittest

import torch

from evaluate_psi_univ_module import _macro_f1, _target_indices, summarize_result, BatchResult


class EvaluatePsiUnivModuleTest(unittest.TestCase):
    def test_target_indices_from_one_hot(self):
        labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        target = _target_indices(labels)
        self.assertTrue(torch.equal(target, torch.tensor([0, 1, 2])))

    def test_macro_f1_perfect_prediction(self):
        pred = torch.tensor([0, 1, 2, 0, 1, 2])
        target = torch.tensor([0, 1, 2, 0, 1, 2])
        self.assertEqual(_macro_f1(pred, target), 1.0)

    def test_summarize_result_returns_expected_metrics(self):
        logits = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        report = summarize_result("toy", BatchResult(logits=logits, labels=labels))
        self.assertEqual(report["name"], "toy")
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
