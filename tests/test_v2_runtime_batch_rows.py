from __future__ import annotations

import unittest

import torch

from bra_universal_plugin import UniversalCandidate, UniversalObservation, UniversalPluginConfig, UniversalPluginOutput
from uniground_v2.runtime import RetrievalResult, TriggerDecision, UniGroundV2LogitsProcessor


class _Tokenizer:
    TOKEN_MAP = {
        "yes": 0,
        " yes": 0,
        "no": 1,
        " no": 1,
    }

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return f"prefix-{ids[0]}"

    def encode(self, text, add_special_tokens=False):
        if text in self.TOKEN_MAP:
            return [self.TOKEN_MAP[text]]
        return [99, 100]


class _AlwaysTrigger:
    def decide(self, scores, input_ids, tokenizer):
        return TriggerDecision(fire=True, reason="entropy_high", entropy=2.0, margin=0.1, content_ratio=1.0)


class _CandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text):
        ids = top_ids.tolist()
        return [
            UniversalCandidate(ids[0], "tok1", "dog", "dog", True),
            UniversalCandidate(ids[1], "tok2", "cat", "cat", True),
        ]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class _Encoder:
    def encode_texts(self, texts):
        return torch.ones((len(texts), 4), dtype=torch.float32)


class _Retriever:
    def retrieve(self, observation, candidates, device):
        return RetrievalResult(
            observation=UniversalObservation(
                image_embedding=observation.image_embedding.to(device),
                region_embeddings=None,
                metadata=dict(observation.metadata),
            ),
            metadata={"region_mode": observation.metadata.get("region_mode", "none")},
        )


class _Scorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        if observation.metadata.get("row") == 0:
            support = torch.tensor([3.0, -3.0], device=candidate_embeddings.device)
            contradiction = torch.tensor([-3.0, 3.0], device=candidate_embeddings.device)
        else:
            support = torch.tensor([-3.0, 3.0], device=candidate_embeddings.device)
            contradiction = torch.tensor([3.0, -3.0], device=candidate_embeddings.device)
        return UniversalPluginOutput(
            support=support,
            contradiction=contradiction,
            abstain=torch.tensor([-8.0, -8.0], device=candidate_embeddings.device),
        )


class _VerifierFavorYesScorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        return UniversalPluginOutput(
            support=torch.tensor([2.0, -2.0], device=candidate_embeddings.device),
            contradiction=torch.tensor([-2.0, 2.0], device=candidate_embeddings.device),
            abstain=torch.tensor([-8.0, -8.0], device=candidate_embeddings.device),
        )


class _WeakVerifierScorer:
    def __call__(self, observation, candidate_embeddings, prefix_embedding):
        return UniversalPluginOutput(
            support=torch.tensor([0.2, -0.2], device=candidate_embeddings.device),
            contradiction=torch.tensor([-0.2, 0.2], device=candidate_embeddings.device),
            abstain=torch.tensor([-8.0, -8.0], device=candidate_embeddings.device),
        )


class _SparseBinaryCandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text, context=None):
        ids = top_ids.tolist()
        return [
            UniversalCandidate(ids[0], "tok0", "filler0", "filler0", False, metadata={"binary_label": ""}),
            UniversalCandidate(ids[1], "tok1", "filler1", "filler1", False, metadata={"binary_label": ""}),
            UniversalCandidate(ids[2], "tok2", "a photo containing dog", "a photo containing dog", True, metadata={"binary_label": "yes"}),
            UniversalCandidate(ids[3], "tok3", "a photo without dog", "a photo without dog", True, metadata={"binary_label": "no"}),
        ]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class _BinaryLeadingCandidateBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text, context=None):
        ids = top_ids.tolist()
        return [
            UniversalCandidate(ids[0], "tok_yes", "a photo containing dog", "a photo containing dog", True, metadata={"binary_label": "yes"}),
            UniversalCandidate(ids[1], "tok_no", "a photo without dog", "a photo without dog", True, metadata={"binary_label": "no"}),
            UniversalCandidate(ids[2], "tok_f0", "filler0", "filler0", False, metadata={"binary_label": ""}),
            UniversalCandidate(ids[3], "tok_f1", "filler1", "filler1", False, metadata={"binary_label": ""}),
        ]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class _NonBinaryGroundableBuilder:
    def build(self, tokenizer, top_ids, gate, prefix_text, context=None):
        ids = top_ids.tolist()
        return [
            UniversalCandidate(ids[0], "tok_alpha", "dog", "dog", True, metadata={"binary_label": ""}),
            UniversalCandidate(ids[1], "tok_beta", "cat", "cat", True, metadata={"binary_label": ""}),
        ]

    def scatter_bias(self, scores, top_ids, bias):
        updated = scores.clone()
        updated[0].scatter_add_(0, top_ids, bias.to(scores.dtype))
        return updated


class RuntimeBatchRowsTest(unittest.TestCase):
    def test_processor_updates_each_batch_row_independently(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=2, bias_scale=0.5),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=[
                UniversalObservation(image_embedding=torch.zeros(4), metadata={"row": 0}),
                UniversalObservation(image_embedding=torch.zeros(4), metadata={"row": 1}),
            ],
            trigger=_AlwaysTrigger(),
            candidate_builder=_CandidateBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        scores = torch.tensor(
            [
                [0.1, 0.9, 0.8, 0.2],
                [0.1, 0.9, 0.8, 0.2],
            ],
            dtype=torch.float32,
        )

        updated = processor(input_ids, scores)

        self.assertGreater(float(updated[0, 1]), float(scores[0, 1]))
        self.assertLess(float(updated[0, 2]), float(scores[0, 2]))
        self.assertLess(float(updated[1, 1]), float(scores[1, 1]))
        self.assertGreater(float(updated[1, 2]), float(scores[1, 2]))
        self.assertEqual(len(processor.get_audit_log()), 2)

    def test_verifier_decision_audit_uses_active_candidate_indices(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=4, bias_scale=0.5),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=UniversalObservation(
                image_embedding=torch.zeros(4),
                metadata={
                    "runtime_context": {
                        "task_name": "pope",
                        "controller_mode": "verifier",
                        "decision_mode": "answer_labels",
                        "answer_mode": "yes_no",
                        "answer_labels": ["yes", "no"],
                        "answer_choice_texts": {"yes": ["yes"], "no": ["no"]},
                        "hypothesis_text_by_label": {
                            "yes": "a photo containing dog",
                            "no": "a photo without dog",
                        },
                    }
                },
            ),
            trigger=_AlwaysTrigger(),
            candidate_builder=_SparseBinaryCandidateBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, 0.2, 0.9, 0.8, 0.0]], dtype=torch.float32)

        updated = processor(input_ids, scores)

        self.assertEqual(tuple(updated.shape), tuple(scores.shape))
        audit = processor.get_audit_log()[0]["decision_audit"]
        self.assertEqual(audit["host_choice"], "no")
        self.assertIn("support_sigmoid", audit["binary_candidates"]["yes"])

    def test_verifier_asymmetry_pushes_yes_when_host_prefers_no(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=4, bias_scale=0.5),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_VerifierFavorYesScorer(),
            observation=UniversalObservation(
                image_embedding=torch.zeros(4),
                metadata={
                    "runtime_context": {
                        "task_name": "pope",
                        "controller_mode": "verifier",
                        "decision_mode": "answer_labels",
                        "answer_mode": "yes_no",
                        "answer_labels": ["yes", "no"],
                        "answer_choice_texts": {"yes": ["yes"], "no": ["no"]},
                        "hypothesis_text_by_label": {
                            "yes": "a photo containing dog",
                            "no": "a photo without dog",
                        },
                    }
                },
            ),
            trigger=_AlwaysTrigger(),
            candidate_builder=_BinaryLeadingCandidateBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.4, 0.9, 0.1, 0.0, -1.0]], dtype=torch.float32)

        updated = processor(input_ids, scores)

        self.assertGreater(float(updated[0, 0]), float(scores[0, 0]))
        self.assertLess(float(updated[0, 1]), float(scores[0, 1]))
        audit = processor.get_audit_log()[0]["decision_audit"]
        self.assertEqual(audit["host_choice"], "no")

    def test_verifier_mode_does_not_fallback_to_generic_bias_without_binary_pair(self):
        class _NoAnswerTokenizer(_Tokenizer):
            def encode(self, text, add_special_tokens=False):
                return [99, 100]

        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=2, bias_scale=0.5),
            tokenizer=_NoAnswerTokenizer(),
            encoder=_Encoder(),
            scorer=_Scorer(),
            observation=UniversalObservation(
                image_embedding=torch.zeros(4),
                metadata={
                    "runtime_context": {
                        "task_name": "pope",
                        "controller_mode": "verifier",
                        "decision_mode": "answer_labels",
                        "answer_mode": "yes_no",
                        "answer_labels": ["yes", "no"],
                        "answer_choice_texts": {"yes": ["yes"], "no": ["no"]},
                        "hypothesis_text_by_label": {
                            "yes": "a photo containing dog",
                            "no": "a photo without dog",
                        },
                    }
                },
            ),
            trigger=_AlwaysTrigger(),
            candidate_builder=_NonBinaryGroundableBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.8, 0.7, 0.1]], dtype=torch.float32)

        updated = processor(input_ids, scores)

        self.assertTrue(torch.equal(updated, scores))
        audit = processor.get_audit_log()[0]["decision_audit"]
        self.assertEqual(audit["evidence_gate_reason"], "missing_answer_token_ids")
        self.assertFalse(audit["evidence_gate_passed"])

    def test_verifier_backoff_when_evidence_delta_is_too_small(self):
        processor = UniGroundV2LogitsProcessor(
            config=UniversalPluginConfig(top_k=4, bias_scale=0.5),
            tokenizer=_Tokenizer(),
            encoder=_Encoder(),
            scorer=_WeakVerifierScorer(),
            observation=UniversalObservation(
                image_embedding=torch.zeros(4),
                metadata={
                    "runtime_context": {
                        "task_name": "pope",
                        "controller_mode": "verifier",
                        "decision_mode": "answer_labels",
                        "answer_mode": "yes_no",
                        "answer_labels": ["yes", "no"],
                        "answer_choice_texts": {"yes": ["yes"], "no": ["no"]},
                        "hypothesis_text_by_label": {
                            "yes": "a photo containing dog",
                            "no": "a photo without dog",
                        },
                        "pope_min_verifier_delta": 0.25,
                        "pope_min_evidence_confidence": 0.05,
                    }
                },
            ),
            trigger=_AlwaysTrigger(),
            candidate_builder=_BinaryLeadingCandidateBuilder(),
            retriever=_Retriever(),
        )
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.4, 0.9, 0.1, 0.0, -1.0]], dtype=torch.float32)

        updated = processor(input_ids, scores)

        self.assertTrue(torch.equal(updated, scores))
        audit = processor.get_audit_log()[0]["decision_audit"]
        self.assertEqual(audit["evidence_gate_reason"], "delta_below_threshold")
        self.assertFalse(audit["evidence_gate_passed"])


if __name__ == "__main__":
    unittest.main()
