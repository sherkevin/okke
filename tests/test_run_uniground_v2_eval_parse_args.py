from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import run_uniground_v2_eval


class RunUniGroundV2EvalParseArgsTest(unittest.TestCase):
    def test_accepts_llava_model(self):
        argv = [
            "run_uniground_v2_eval.py",
            "--model",
            "llava-v1.5-7b",
            "--dataset",
            "pope",
            "--pope_controller_mode",
            "frontier",
            "--pope_min_verifier_delta",
            "0.07",
            "--pope_min_evidence_confidence",
            "0.04",
        ]
        with patch.object(sys, "argv", argv):
            args = run_uniground_v2_eval.parse_args()
        self.assertEqual(args.model, "llava-v1.5-7b")
        self.assertEqual(args.pope_controller_mode, "frontier")
        self.assertEqual(args.pope_min_verifier_delta, 0.07)
        self.assertEqual(args.pope_min_evidence_confidence, 0.04)


if __name__ == "__main__":
    unittest.main()
