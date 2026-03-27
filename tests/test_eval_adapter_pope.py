from __future__ import annotations

import unittest

from uniground_v2.task_adapter import PopeTaskAdapter


class EvalAdapterPopeTest(unittest.TestCase):
    def test_pope_adapter_formats_and_parses(self):
        adapter = PopeTaskAdapter()
        question = adapter.format_question("Is there a dog in the image")
        self.assertIn("yes or no", question.lower())
        self.assertEqual(adapter.parse_prediction("Yes"), "yes")
        self.assertEqual(adapter.parse_prediction("no"), "no")


if __name__ == "__main__":
    unittest.main()
