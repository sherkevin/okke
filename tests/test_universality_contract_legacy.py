from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from validate_uniground_universality import (
    EXPECTED_FORMAT,
    EXPECTED_MODULE,
    EXPECTED_ROUTE,
    EXPECTED_TEXT_ENCODER,
    EXPECTED_VISION_ENCODER,
    validate_checkpoint,
    validate_result,
)


class LegacyUniversalityContractTest(unittest.TestCase):
    def test_synthetic_checkpoint_and_result_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "psi_univ_legacy.pt"
            result_path = root / "result.json"

            checkpoint_payload = {
                "state_dict": {},
                "embed_dim": 768,
                "contract": {
                    "checkpoint_format": EXPECTED_FORMAT,
                    "plugin_route": EXPECTED_ROUTE,
                    "learned_module": EXPECTED_MODULE,
                    "frozen_vision_encoder_name": EXPECTED_VISION_ENCODER,
                    "frozen_text_encoder_name": EXPECTED_TEXT_ENCODER,
                    "embedding_dim": 768,
                    "region_features_enabled": False,
                    "uses_model_native_hidden_states": False,
                    "uses_lm_head_geometry": False,
                    "learned_per_model_adapter": False,
                    "parameter_free_tokenizer_bridge": True,
                    "source_hashes": {"synthetic": "ok"},
                },
            }
            torch.save(checkpoint_payload, checkpoint_path)
            ck_ok, _, ck_info = validate_checkpoint(checkpoint_path)
            self.assertTrue(ck_ok)

            result_payload = {
                "universal_claim_manifest": {
                    "method_route": EXPECTED_ROUTE,
                    "learned_module": EXPECTED_MODULE,
                    "uses_model_native_hidden_states": False,
                    "uses_lm_head_geometry": False,
                    "learned_per_model_adapter": False,
                    "parameter_free_tokenizer_bridge": True,
                    "abstention_controls_behavior": True,
                    "psi_univ_checkpoint": {
                        "checkpoint_sha256": ck_info["checkpoint_sha256"],
                    },
                },
                "prefix_ambiguity_rate": 0.0,
                "span_collapse_errors": 0.0,
                "suffix_stability_rate": 1.0,
                "abstention_rate": 0.0,
                "abort_trigger_rate": 0.0,
                "abort_backoff_verified_steps": 0.0,
                "latency_split": {
                    "candidate_construction_ms": 0.1,
                    "sidecar_scoring_ms": 0.2,
                    "bridge_redistribution_ms": 0.1,
                    "jitter_ms": 0.0,
                },
            }
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")
            result_ok, _, _ = validate_result(result_path, ck_info)
            self.assertTrue(result_ok)


if __name__ == "__main__":
    unittest.main()
