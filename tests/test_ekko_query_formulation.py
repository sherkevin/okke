from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EKKO"))

from chord.knowledge_kernel_evaluator import score_anchor_relevance  # noqa: E402
from chord.query_formulation import build_model_query, extract_anchor_query  # noqa: E402


class EkkoQueryFormulationTests(unittest.TestCase):
    def test_extract_anchor_query_recovers_object_phrase(self) -> None:
        self.assertEqual(
            extract_anchor_query("Is there a snowboard in the image?"),
            "snowboard",
        )
        self.assertEqual(
            extract_anchor_query("Is there a dining table in the image?"),
            "dining table",
        )

    def test_build_model_query_appends_suffix_only_for_model_side(self) -> None:
        formatted = build_model_query(
            "Is there a snowboard in the image?",
            suffix="Answer with exactly one word: yes or no.",
        )
        self.assertEqual(
            formatted,
            "Is there a snowboard in the image? Answer with exactly one word: yes or no.",
        )

    def test_score_anchor_relevance_handles_plural_and_alias_matches(self) -> None:
        self.assertGreaterEqual(score_anchor_relevance("skis", "ski"), 0.9)
        self.assertGreaterEqual(score_anchor_relevance("couch", "sofa"), 0.9)
        self.assertGreaterEqual(score_anchor_relevance("cell phone", "phone"), 0.9)
        self.assertLess(score_anchor_relevance("snowboard", "car"), 0.2)


if __name__ == "__main__":
    unittest.main()
