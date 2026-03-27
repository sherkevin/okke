from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from train_universal_plugin_v2 import (
    CHECKPOINT_FORMAT,
    CONTRACT_VERSION,
    build_contract_v2,
)


class CheckpointContractV2Test(unittest.TestCase):
    def test_contract_contains_v2_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features = Path(tmpdir) / "features.pt"
            torch.save({"dummy": True}, features)
            args = argparse.Namespace(
                vision_encoder_name="openai/clip-vit-large-patch14::image",
                text_encoder_name="openai/clip-vit-large-patch14::text",
                region_mode="retrieved_topr",
            )
            contract = build_contract_v2(
                args=args,
                embed_dim=768,
                features_path=features,
                payload_metadata={"record_count": 4},
            )
            self.assertEqual(contract["contract_version"], CONTRACT_VERSION)
            self.assertEqual(contract["checkpoint_format"], CHECKPOINT_FORMAT)
            self.assertTrue(contract["region_features_enabled"])
            self.assertEqual(contract["region_mode"], "retrieved_topr")


if __name__ == "__main__":
    unittest.main()
