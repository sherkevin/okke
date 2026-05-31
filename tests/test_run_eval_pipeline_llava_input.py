from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from run_eval_pipeline import build_single_input


class _LegacyLlavaProcessor:
    def __call__(self, images, text, return_tensors="pt"):
        self.last_text = text
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        }


class RunEvalPipelineLlavaInputTest(unittest.TestCase):
    def test_build_single_input_falls_back_for_legacy_llava_processor(self):
        processor = _LegacyLlavaProcessor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.jpg"
            Image.new("RGB", (8, 8), color="white").save(image_path)
            inputs = build_single_input(processor, str(image_path), "Is there a dog?", device="cpu")
        self.assertEqual(processor.last_text, "USER: <image>\nIs there a dog? ASSISTANT:")
        self.assertEqual(tuple(inputs["input_ids"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
