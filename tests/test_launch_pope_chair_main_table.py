from __future__ import annotations

import unittest

from launch_pope_chair_main_table import build_jobs, launch_script


class LaunchPopeChairMainTableTest(unittest.TestCase):
    def test_build_jobs_contains_expected_main_table_runs(self):
        jobs = build_jobs(model="qwen3-vl-8b", pope_count=10, chair_count=5, include_internal_controls=False)
        names = {job.name for job in jobs}
        self.assertIn("pope_baselines", names)
        self.assertIn("pope_uniground_family", names)
        self.assertIn("chair_opera", names)
        self.assertIn("chair_uniground_family", names)

    def test_launch_script_is_detached_and_uses_single_gpu(self):
        jobs = build_jobs(model="qwen3-vl-8b", pope_count=10, chair_count=5, include_internal_controls=False)

        class _Args:
            model = "qwen3-vl-8b"
            gpu = 1

        script, runner, log = launch_script(_Args(), jobs)
        self.assertIn("nohup /bin/bash", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=1", script)
        self.assertIn("run_eval_pipeline.py", script)
        self.assertIn("run_uniground_eval.py", script)
        self.assertTrue(runner.endswith(".sh"))
        self.assertTrue(log.endswith(".log"))


if __name__ == "__main__":
    unittest.main()
