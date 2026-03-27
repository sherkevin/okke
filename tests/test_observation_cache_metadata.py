from __future__ import annotations

import unittest

import torch
from PIL import Image

from uniground_v2.observation import build_v2_observation_cache
from uniground_v2.regions import RegionProposal


class _Encoder:
    def encode_image(self, image):
        width, height = image.size
        return torch.tensor([float(width), float(height)], dtype=torch.float32)


class ObservationCacheMetadataTest(unittest.TestCase):
    def test_cache_records_pre_generate_metadata_for_grid_regions(self):
        image = Image.new("RGB", (40, 20), color=(255, 255, 255))
        result = build_v2_observation_cache(_Encoder(), image, mode="grid_regions")
        self.assertEqual(result.cache_info["cache_stage"], "pre_generate")
        self.assertEqual(result.cache_info["region_mode"], "grid_regions")
        self.assertEqual(result.cache_info["region_count"], 4)
        self.assertFalse(result.cache_info["uses_detector_proposals"])
        self.assertIn("precompute_ms", result.observation.metadata)

    def test_cache_records_detector_metadata_when_proposals_are_provided(self):
        image = Image.new("RGB", (40, 20), color=(255, 255, 255))
        result = build_v2_observation_cache(
            _Encoder(),
            image,
            proposals=[RegionProposal(0, 0, 20, 20, score=0.9, label="dog")],
            mode="detector_regions",
        )
        self.assertEqual(result.cache_info["region_mode"], "detector_regions")
        self.assertEqual(result.cache_info["region_count"], 1)
        self.assertTrue(result.cache_info["uses_detector_proposals"])


if __name__ == "__main__":
    unittest.main()
