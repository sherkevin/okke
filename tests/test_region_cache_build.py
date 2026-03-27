from __future__ import annotations

import unittest

import torch
from PIL import Image

from uniground_v2.regions import RegionProposal, RegionProposalCacheBuilder


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


class RegionCacheBuildTest(unittest.TestCase):
    def test_detector_style_proposals_build_region_cache(self):
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))
        builder = RegionProposalCacheBuilder(_Encoder())
        observation = builder.build(
            image=image,
            proposals=[
                RegionProposal(0, 0, 50, 40, score=0.9, label="dog"),
                RegionProposal(50, 40, 100, 80, score=0.8, label="cat"),
            ],
        )
        self.assertEqual(observation.region_embeddings.shape, (2, 2))
        self.assertEqual(observation.metadata["region_mode"], "detector_regions")
        self.assertEqual(len(observation.metadata["region_boxes"]), 2)

    def test_detector_style_proposals_use_batch_image_encoding_when_available(self):
        image = Image.new("RGB", (100, 80), color=(255, 255, 255))
        encoder = _BatchEncoder()
        builder = RegionProposalCacheBuilder(encoder)
        observation = builder.build(
            image=image,
            proposals=[
                RegionProposal(0, 0, 50, 40, score=0.9, label="dog"),
                RegionProposal(50, 40, 100, 80, score=0.8, label="cat"),
            ],
        )
        self.assertEqual(observation.region_embeddings.shape, (2, 2))
        self.assertEqual(encoder.batch_calls, 1)
        self.assertEqual(encoder.single_calls, 3)


if __name__ == "__main__":
    unittest.main()
