from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from run_uniground_v2_eval import build_single_input


class _LlavaProcessor:
    def apply_chat_template(self, conversation, add_generation_prompt=True):
        assert conversation[0]["content"][0]["type"] == "image"
        return "PROMPT"

    def __call__(self, images, text, return_tensors="pt"):
        assert text == "PROMPT"
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        }


class RunUniGroundV2EvalLlavaInputTest(unittest.TestCase):
    def test_build_single_input_uses_llava_conversation_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.jpg"
            Image.new("RGB", (8, 8), color="white").save(image_path)
            image, inputs = build_single_input(_LlavaProcessor(), str(image_path), "Is there a dog?", device="cpu")
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(tuple(inputs["input_ids"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
