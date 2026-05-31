from __future__ import annotations

import unittest

from uniground_v2.bridge import MorphBoundaryResolver, SubwordBridgeState


class BridgePunctuationSkipTest(unittest.TestCase):
    def test_punctuation_is_skipped(self):
        resolver = MorphBoundaryResolver()
        state = SubwordBridgeState()
        resolution = resolver.resolve(state, ".")
        self.assertEqual(resolution.status, "skip")


if __name__ == "__main__":
    unittest.main()
