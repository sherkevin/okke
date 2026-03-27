from __future__ import annotations

import unittest

from uniground_v2.bridge import MorphBoundaryResolver, SubwordBridgeState


class BridgePendingStateTest(unittest.TestCase):
    def test_short_fragment_is_marked_pending(self):
        resolver = MorphBoundaryResolver()
        state = SubwordBridgeState()
        resolution = resolver.resolve(state, "ju")
        self.assertEqual(resolution.status, "pending")
        self.assertEqual(state.pending_text, "ju")


if __name__ == "__main__":
    unittest.main()
