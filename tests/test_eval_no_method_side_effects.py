from __future__ import annotations

import unittest
from pathlib import Path


class EvalNoMethodSideEffectsTest(unittest.TestCase):
    def test_v2_runner_does_not_contain_legacy_fastpath_heuristics(self):
        runner = Path(__file__).resolve().parents[1] / "run_uniground_v2_eval.py"
        text = runner.read_text(encoding="utf-8")
        self.assertNotIn("evidence_bonus", text)
        self.assertNotIn("read_object_evidence", text)
        self.assertNotIn("score_binary_choice", text)


if __name__ == "__main__":
    unittest.main()
