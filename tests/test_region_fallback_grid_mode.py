from __future__ import annotations

import unittest

import torch
from PIL import Image

from uniground_v2.regions import RegionProposalCacheBuilder


class _Encoder:
    def encode_image(self, image):
        width, height = image.size
        return torch.tensor([float(width), float(height)], dtype=torch.float32)


class _BatchEncoder:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0

    def encode_image(self, image):
        self.single_calls += 1
        width, height = image.size
        return torch.tensor([float(width), float(height)], dtype=torch.float32)

    def encode_images(self, images):
        self.batch_calls += 1
        return torch.stack([self.encode_image(image) for image in images], dim=0)


class RegionFallbackGridModeTest(unittest.TestCase):
    def test_grid_fallback_builds_regions_when_no_proposals(self):
        image = Image.new("RGB", (120, 80), color=(128, 128, 128))
        builder = RegionProposalCacheBuilder(_Encoder(), grid_size=2)
        observation = builder.build(image=image, proposals=None, mode="grid_regions")
        self.assertEqual(observation.region_embeddings.shape, (4, 2))
        self.assertEqual(observation.metadata["region_mode"], "grid_regions")
        self.assertEqual(len(observation.metadata["region_boxes"]), 4)

    def test_grid_fallback_uses_batch_image_encoding_when_available(self):
        image = Image.new("RGB", (120, 80), color=(128, 128, 128))
        encoder = _BatchEncoder()
        builder = RegionProposalCacheBuilder(encoder, grid_size=2)
        observation = builder.build(image=image, proposals=None, mode="grid_regions")
        self.assertEqual(observation.region_embeddings.shape, (4, 2))
        self.assertEqual(encoder.batch_calls, 1)
        self.assertEqual(encoder.single_calls, 5)


if __name__ == "__main__":
    unittest.main()
