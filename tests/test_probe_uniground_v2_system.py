from __future__ import annotations

import unittest

from probe_uniground_v2_system import extract_paths


class ProbeUniGroundV2SystemTest(unittest.TestCase):
    def test_extract_paths_parses_all_markers(self):
        text = "\n".join(
            [
                "HARDCODED_POPE_JSON=/tmp/a.json",
                "CHECKPOINT_POPE_JSON=/tmp/b.json",
                "HARDCODED_CHAIR_JSON=/tmp/c.json",
                "CHECKPOINT_CHAIR_JSON=/tmp/d.json",
            ]
        )
        paths = extract_paths(text)
        self.assertEqual(paths["HARDCODED_POPE_JSON"], "/tmp/a.json")
        self.assertEqual(paths["CHECKPOINT_CHAIR_JSON"], "/tmp/d.json")


if __name__ == "__main__":
    unittest.main()
