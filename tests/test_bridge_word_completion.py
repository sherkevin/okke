from __future__ import annotations

import unittest

from uniground_v2.bridge import MorphBoundaryResolver, SubwordBridgeState


class BridgeWordCompletionTest(unittest.TestCase):
    def test_continuation_fragment_completes_pending_word(self):
        resolver = MorphBoundaryResolver()
        state = SubwordBridgeState()
        resolver.resolve(state, "ju")
        resolution = resolver.resolve(state, "##mp")
        self.assertEqual(resolution.status, "complete")
        self.assertEqual(resolution.span_text, "jump")
        self.assertEqual(state.pending_text, "")


if __name__ == "__main__":
    unittest.main()
