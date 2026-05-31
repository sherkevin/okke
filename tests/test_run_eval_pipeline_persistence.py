from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_eval_pipeline


class RunEvalPipelinePersistenceTest(unittest.TestCase):
    def test_resolve_output_json_path_prefers_explicit_path(self):
        args = type(
            "Args",
            (),
            {
                "output_json": "/tmp/custom.json",
                "run_id": None,
                "method": "ifcb",
                "dataset": "pope",
                "pope_split": "random",
            },
        )()
        out_path = run_eval_pipeline._resolve_output_json_path(args)
        self.assertEqual(out_path, Path("/tmp/custom.json"))

    def test_resolve_output_json_path_uses_run_id_and_pope_split(self):
        args = type(
            "Args",
            (),
            {
                "output_json": None,
                "run_id": "gpu1-fullrun",
                "method": "ifcb",
                "dataset": "pope",
                "pope_split": "popular",
            },
        )()
        out_path = run_eval_pipeline._resolve_output_json_path(args)
        self.assertEqual(out_path, run_eval_pipeline.LOG_DIR / "ifcb_pope_popular_gpu1-fullrun.json")

    def test_persist_run_snapshot_writes_status_and_progress(self):
        args = type(
            "Args",
            (),
            {
                "dataset": "pope",
                "method": "ifcb",
                "model": "llava-v1.5-7b",
                "pope_split": "adversarial",
                "run_id": "gpu1-fullrun",
            },
        )()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "snapshot.json"
            run_eval_pipeline._persist_run_snapshot(
                args,
                output_path=out_path,
                status="partial",
                metrics={"accuracy": 0.75, "sample_count": 12},
                completed_samples=12,
                target_samples=3000,
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["completed_samples"], 12)
        self.assertEqual(payload["target_samples"], 3000)
        self.assertEqual(payload["pope_split"], "adversarial")
        self.assertEqual(payload["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
