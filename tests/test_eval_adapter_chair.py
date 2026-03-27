from __future__ import annotations

import unittest

from uniground_v2.task_adapter import ChairTaskAdapter


class EvalAdapterChairTest(unittest.TestCase):
    def test_chair_adapter_returns_prompt(self):
        adapter = ChairTaskAdapter(prompt="Describe the image.")
        self.assertEqual(adapter.format_question(), "Describe the image.")


if __name__ == "__main__":
    unittest.main()
