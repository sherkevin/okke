from __future__ import annotations

import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class LegacyRuntimeBoundarySmokeTest(unittest.TestCase):
    def test_legacy_entrypoints_are_marked(self) -> None:
        eval_text = (PROJECT / "run_uniground_eval.py").read_text(encoding="utf-8")
        runtime_text = (PROJECT / "uniground_runtime.py").read_text(encoding="utf-8")
        plugin_text = (PROJECT / "bra_universal_plugin.py").read_text(encoding="utf-8")

        self.assertIn("Legacy UniGround v1 evaluation entry", eval_text)
        self.assertIn("Legacy UniGround v1 runtime", runtime_text)
        self.assertIn("v2 refactor", plugin_text)

    def test_migration_boundary_note_exists(self) -> None:
        note = PROJECT / "UNIGROUND_V2_MIGRATION_BOUNDARY.md"
        self.assertTrue(note.exists(), "migration boundary note must exist")
        text = note.read_text(encoding="utf-8")
        self.assertIn("Legacy v1 Files", text)
        self.assertIn("V2 Development Rule", text)


if __name__ == "__main__":
    unittest.main()
