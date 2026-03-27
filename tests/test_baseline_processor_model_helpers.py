from __future__ import annotations

import unittest

from baseline_processors import _get_decoder_layers, _get_lm_head, _get_num_hidden_layers


class _Layers(list):
    pass


class _QwenInner:
    def __init__(self):
        self.language_model = type("LM", (), {"layers": _Layers([1, 2, 3])})()


class _QwenModel:
    def __init__(self):
        self.model = _QwenInner()
        self.lm_head = "qwen_lm_head"
        self.config = type("Cfg", (), {"text_config": type("TextCfg", (), {"num_hidden_layers": 3})()})()


class _LlavaLanguageModel:
    def __init__(self):
        self.model = type("DecoderRoot", (), {"layers": _Layers([1, 2, 3, 4])})()
        self.lm_head = "llava_lm_head"
        self.config = type("Cfg", (), {"num_hidden_layers": 4})()


class _LlavaModel:
    def __init__(self):
        self.language_model = _LlavaLanguageModel()


class _DecoderOnlyRoot:
    def __init__(self):
        self.decoder = type("Decoder", (), {"layers": _Layers([1, 2, 3, 4, 5])})()


class _InstructBlipLanguageModel:
    def __init__(self):
        self.model = _DecoderOnlyRoot()
        self.lm_head = "ib_lm_head"
        self.config = type("Cfg", (), {"num_hidden_layers": 5})()


class _InstructBlipModel:
    def __init__(self):
        self.language_model = _InstructBlipLanguageModel()


class BaselineProcessorModelHelperTest(unittest.TestCase):
    def test_qwen_helpers(self):
        model = _QwenModel()
        self.assertEqual(len(_get_decoder_layers(model)), 3)
        self.assertEqual(_get_lm_head(model), "qwen_lm_head")
        self.assertEqual(_get_num_hidden_layers(model), 3)

    def test_llava_helpers(self):
        model = _LlavaModel()
        self.assertEqual(len(_get_decoder_layers(model)), 4)
        self.assertEqual(_get_lm_head(model), "llava_lm_head")
        self.assertEqual(_get_num_hidden_layers(model), 4)

    def test_instructblip_helpers(self):
        model = _InstructBlipModel()
        self.assertEqual(len(_get_decoder_layers(model)), 5)
        self.assertEqual(_get_lm_head(model), "ib_lm_head")
        self.assertEqual(_get_num_hidden_layers(model), 5)


if __name__ == "__main__":
    unittest.main()
