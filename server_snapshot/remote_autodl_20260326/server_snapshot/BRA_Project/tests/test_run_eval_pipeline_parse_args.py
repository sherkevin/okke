from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import run_eval_pipeline


class RunEvalPipelineParseArgsTest(unittest.TestCase):
    def test_defaults_to_ifcb_method(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "llava-v1.5-7b",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.method, "ifcb")

    def test_accepts_new_model_and_beam_search_method(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "llava-v1.5-7b",
            "--dataset",
            "pope",
            "--method",
            "beam_search",
            "--mini_test",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.model, "llava-v1.5-7b")
        self.assertEqual(args.method, "beam_search")

    def test_accepts_damo_method(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "llava-v1.5-7b",
            "--dataset",
            "pope",
            "--method",
            "damo",
            "--mini_test",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.method, "damo")

    def test_accepts_ifcb_method(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "llava-v1.5-7b",
            "--dataset",
            "chair",
            "--method",
            "ifcb",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.dataset, "chair")
        self.assertEqual(args.method, "ifcb")

    def test_accepts_new_theory_datasets(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "instructblip-7b",
            "--dataset",
            "hallusionbench",
            "--method",
            "ifcb",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.model, "instructblip-7b")
        self.assertEqual(args.dataset, "hallusionbench")

    def test_accepts_output_json_run_id_and_checkpoint_every(self):
        argv = [
            "run_eval_pipeline.py",
            "--model",
            "llava-v1.5-7b",
            "--dataset",
            "pope",
            "--method",
            "ifcb",
            "--output_json",
            "/tmp/ifcb_random.json",
            "--run_id",
            "gpu1-fullrun",
            "--checkpoint_every",
            "25",
        ]
        with patch.object(sys, "argv", argv):
            args = run_eval_pipeline.parse_args()
        self.assertEqual(args.output_json, "/tmp/ifcb_random.json")
        self.assertEqual(args.run_id, "gpu1-fullrun")
        self.assertEqual(args.checkpoint_every, 25)


if __name__ == "__main__":
    unittest.main()
