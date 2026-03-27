from __future__ import annotations

import unittest

import torch

from ifcb.processor import (
    IFCBConfig,
    InstructBLIPIFCBProcessor,
    LLaVAIFCBProcessor,
    apply_ifcb_penalty,
    build_modal_masks,
    build_query_token_masks,
    create_ifcb_processor,
    compute_commitment_risks,
    is_semantic_token,
    resolve_probe_layers,
)


class _FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.mapping.get(int(token_id), f"<{token_id}>") for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        reverse = {value: key for key, value in self.mapping.items()}
        if text in reverse:
            return [reverse[text]]
        return []


class IFCBHelperTest(unittest.TestCase):
    def test_resolve_probe_layers_excludes_final_layer(self):
        self.assertEqual(resolve_probe_layers(6), [1, 2, 3, 4])

    def test_apply_ifcb_penalty_only_changes_frontier(self):
        scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        candidate_ids = torch.tensor([1, 2], dtype=torch.long)
        risks = {
            1: {"risk": 0.5},
            2: {"risk": 0.25},
        }
        adjusted = apply_ifcb_penalty(scores, candidate_ids, risks, alpha=2.0)
        self.assertTrue(torch.allclose(adjusted[0, [0, 3]], scores[0, [0, 3]]))
        self.assertAlmostEqual(float(adjusted[0, 1].item()), 2.0)
        self.assertAlmostEqual(float(adjusted[0, 2].item()), 1.5)

    def test_compute_commitment_risks_penalizes_late_nonpersistent_token(self):
        candidate_ids = torch.tensor([1, 2], dtype=torch.long)
        probe_logprobs = [
            torch.log(torch.tensor([[0.80, 0.10, 0.10]], dtype=torch.float32)),
            torch.log(torch.tensor([[0.75, 0.15, 0.10]], dtype=torch.float32)),
        ]
        final_logprobs = torch.log(torch.tensor([[0.10, 0.20, 0.70]], dtype=torch.float32))
        probe_topk_ids = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ]
        visual_participation = {1: 0.8, 2: 0.1}
        risks = compute_commitment_risks(
            candidate_ids=candidate_ids,
            probe_logprobs=probe_logprobs,
            final_logprobs=final_logprobs,
            probe_topk_ids=probe_topk_ids,
            visual_participation=visual_participation,
        )
        self.assertEqual(risks[1]["risk"], 0.0)
        self.assertGreater(risks[2]["late_surge"], 0.0)
        self.assertGreater(risks[2]["risk"], 0.0)

    def test_chair_candidate_filter_skips_punctuation_only_tokens(self):
        tokenizer = _FakeTokenizer({
            0: "<pad>",
            1: ".",
            2: ",",
            3: "dog",
            4: "cat",
            10: "yes",
            11: "no",
        })
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.tokenizer = tokenizer
        proc.cfg = IFCBConfig(frontier_k=5)
        proc._pope_binary_ids = {"yes": [10], "no": [11]}
        scores = torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, -1.0, -1.5]])
        candidate_ids = proc._candidate_ids("chair", scores)
        self.assertEqual(candidate_ids.tolist(), [3, 4])

    def test_pope_candidate_ids_match_binary_answer_surface(self):
        tokenizer = _FakeTokenizer({
            0: "maybe",
            1: "dog",
            2: "cat",
            3: "tree",
            10: "yes",
            11: "no",
        })
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.tokenizer = tokenizer
        proc.cfg = IFCBConfig(frontier_k=4)
        proc._pope_binary_ids = {"yes": [10], "no": [11]}
        proc._choice_ids = {"A": [20], "B": [21], "C": [22], "D": [23]}
        scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 5.0, 4.5]])
        candidate_ids = proc._candidate_ids("pope", scores)
        self.assertEqual(candidate_ids.tolist(), [10, 11])

    def test_pope_selection_keeps_explicit_yes_no_control(self):
        tokenizer = _FakeTokenizer({0: "dog", 1: "cat", 10: "yes", 11: "no"})
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.tokenizer = tokenizer
        proc.cfg = IFCBConfig(frontier_k=4)
        proc._pope_binary_ids = {"yes": [10], "no": [11]}
        adjusted_scores = torch.tensor([[12.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 3.0]])
        next_token = proc._select_next_token("pope", adjusted_scores)
        self.assertEqual(next_token.tolist(), [[10]])

    def test_semantic_token_helper_recognizes_words(self):
        tokenizer = _FakeTokenizer({1: ".", 2: " dog"})
        self.assertFalse(is_semantic_token(tokenizer, 1))
        self.assertTrue(is_semantic_token(tokenizer, 2))

    def test_build_modal_masks_expands_single_image_placeholder(self):
        input_ids = torch.tensor([[101, 32000, 102, 103]], dtype=torch.long)
        visual_mask, text_mask = build_modal_masks(input_ids, image_token_id=32000, hidden_seq_len=8)
        self.assertEqual(visual_mask.tolist(), [False, True, True, True, True, True, False, False])
        self.assertEqual(text_mask.tolist(), [True, False, False, False, False, False, True, True])

    def test_build_query_token_masks_uses_prefix_query_tokens(self):
        visual_mask, text_mask = build_query_token_masks(query_token_count=4, hidden_seq_len=7, device=torch.device("cpu"))
        self.assertEqual(visual_mask.tolist(), [True, True, True, True, False, False, False])
        self.assertEqual(text_mask.tolist(), [False, False, False, False, True, True, True])

    def test_mmbench_candidate_ids_use_choice_tokens(self):
        tokenizer = _FakeTokenizer({
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            10: "yes",
            11: "no",
        })
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.tokenizer = tokenizer
        proc.cfg = IFCBConfig(frontier_k=4)
        proc._pope_binary_ids = {"yes": [10], "no": [11]}
        proc._choice_ids = {"A": [1], "B": [2], "C": [3], "D": [4]}
        scores = torch.tensor([[0.1, 2.0, 1.5, 1.0, 0.9, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0]])
        candidate_ids = proc._candidate_ids("mmbench", scores)
        self.assertEqual(candidate_ids.tolist(), [1, 2, 3, 4])

    def test_fusion_layer_uses_decoder_midpoint(self):
        model = type("Model", (), {"config": type("Cfg", (), {"num_query_tokens": 32})()})()
        tokenizer = _FakeTokenizer({1: "A", 2: "B", 3: "C", 4: "D", 10: "yes", 11: "no"})
        with unittest.mock.patch("ifcb.processor._get_lm_head", return_value=object()), \
             unittest.mock.patch("ifcb.processor._get_decoder_layers", return_value=[object()] * 6), \
             unittest.mock.patch("ifcb.processor._get_decoder_root", return_value=type("Root", (), {"norm": object()})()), \
             unittest.mock.patch("ifcb.processor._LayerCapture", return_value=type("Cap", (), {"remove": lambda self: None})()):
            proc = create_ifcb_processor(model, tokenizer, "instructblip", image_token_id=32001, config=IFCBConfig())
        self.assertEqual(proc.fusion_layer, 3)

    def test_visual_participation_raises_when_grad_path_is_missing(self):
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.model_family = "llava"
        proc.query_token_count = 0
        proc.image_token_id = 32000
        proc.cfg = IFCBConfig()
        proc.fusion_layer = 0
        proc._capture = type("Cap", (), {"hidden_by_layer": {0: torch.zeros((1, 4, 3), dtype=torch.float32)}})()
        with self.assertRaisesRegex(RuntimeError, "gradient-enabled"):
            proc._compute_visual_participation(
                torch.tensor([1], dtype=torch.long),
                torch.tensor([[101, 32000, 102, 103]], dtype=torch.long),
                torch.tensor([[0.1, 0.9, 0.0]], dtype=torch.float32, requires_grad=True),
            )

    def test_apply_ifcb_uses_candidate_surface_width_for_persistence_topk(self):
        proc = LLaVAIFCBProcessor.__new__(LLaVAIFCBProcessor)
        proc.cfg = IFCBConfig(frontier_k=2, probe_topk=5, alpha=1.0)
        proc.probe_layers = [0]
        proc._capture = type(
            "Cap",
            (),
            {
                "hidden_by_layer": {
                    0: torch.tensor([[[4.0, 3.0, 2.0, 1.0]]], dtype=torch.float32, requires_grad=True),
                }
            },
        )()
        proc.lm_head = torch.nn.Identity()
        proc.final_norm = torch.nn.Identity()
        proc._record_step = lambda _candidate_ids, _risks: None
        proc._compute_visual_participation = lambda candidate_ids, input_ids, scores: {
            int(token_id): 0.0 for token_id in candidate_ids.tolist()
        }
        proc.tokenizer = _FakeTokenizer({0: "dog", 1: "cat", 2: "yes", 3: "no"})
        proc._pope_binary_ids = {"yes": [2], "no": [3]}
        proc._choice_ids = {"A": [4], "B": [5], "C": [6], "D": [7]}
        scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32, requires_grad=True)
        observed = {}

        def _fake_compute_commitment_risks(**kwargs):
            observed["topk_width"] = int(kwargs["probe_topk_ids"][0].numel())
            return {int(token_id): {"risk": 0.0, "visual_participation": 0.0} for token_id in kwargs["candidate_ids"].tolist()}

        with unittest.mock.patch("ifcb.processor.compute_commitment_risks", side_effect=_fake_compute_commitment_risks):
            adjusted = proc._apply_ifcb("pope", torch.tensor([[101, 32000, 102]], dtype=torch.long), scores)

        self.assertTrue(torch.allclose(adjusted, scores))
        self.assertEqual(observed["topk_width"], 4)

    def test_create_ifcb_processor_supports_instructblip(self):
        model = type("Model", (), {"config": type("Cfg", (), {"num_query_tokens": 32})()})()
        tokenizer = _FakeTokenizer({1: "A", 2: "B", 3: "C", 4: "D", 10: "yes", 11: "no"})
        with unittest.mock.patch("ifcb.processor._get_lm_head", return_value=object()), \
             unittest.mock.patch("ifcb.processor._get_decoder_layers", return_value=[object(), object(), object(), object()]), \
             unittest.mock.patch("ifcb.processor._get_decoder_root", return_value=type("Root", (), {"norm": object()})()), \
             unittest.mock.patch("ifcb.processor._LayerCapture", return_value=type("Cap", (), {"remove": lambda self: None})()):
            proc = create_ifcb_processor(model, tokenizer, "instructblip", image_token_id=32001, config=IFCBConfig())
        self.assertIsInstance(proc, InstructBLIPIFCBProcessor)


if __name__ == "__main__":
    unittest.main()
