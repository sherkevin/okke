from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bra_universal_plugin import MLPUniversalScorer
from train_universal_plugin_v2 import save_checkpoint_v2


class CheckpointRoundtripV2Test(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_contract_and_dims(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "psi_v2.pt"
            model = MLPUniversalScorer(embed_dim=8, hidden_dim=16)
            contract = {
                "checkpoint_format": "psi_univ_checkpoint_v2",
                "contract_version": "uniground_train_contract_v2",
            }
            save_checkpoint_v2(
                model=model,
                output_path=output,
                embed_dim=8,
                hidden_dim=16,
                contract=contract,
            )
            payload = torch.load(output, map_location="cpu")
            self.assertEqual(payload["embed_dim"], 8)
            self.assertEqual(payload["hidden_dim"], 16)
            self.assertEqual(payload["contract"]["checkpoint_format"], "psi_univ_checkpoint_v2")


if __name__ == "__main__":
    unittest.main()
