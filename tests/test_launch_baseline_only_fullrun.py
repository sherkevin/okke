from __future__ import annotations

import unittest

from launch_baseline_only_fullrun import build_jobs, launch_script


class LaunchBaselineOnlyFullrunTest(unittest.TestCase):
    def test_build_jobs_includes_all_requested_methods(self):
        jobs = build_jobs(
            model="qwen3-vl-2b",
            pope_split="random",
            pope_count=5,
            chair_count=2,
            methods=["base", "vcd"],
            datasets=["pope", "chair"],
        )
        names = [job.name for job in jobs]
        self.assertEqual(names, ["pope_base", "pope_vcd", "chair_base", "chair_vcd"])

    def test_build_jobs_can_limit_to_pope_only(self):
        jobs = build_jobs(
            model="qwen3-vl-2b",
            pope_split="random",
            pope_count=5,
            chair_count=2,
            methods=["base", "vcd"],
            datasets=["pope"],
        )
        self.assertEqual([job.name for job in jobs], ["pope_base", "pope_vcd"])

    def test_launch_script_contains_detached_runner_and_counts(self):
        jobs = build_jobs(
            model="qwen3-vl-2b",
            pope_split="random",
            pope_count=5,
            chair_count=2,
            methods=["base"],
            datasets=["pope", "chair"],
        )

        class _Args:
            model = "qwen3-vl-2b"
            gpu = 1

        script = launch_script(_Args(), jobs)
        self.assertIn("nohup /bin/bash", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", script)
        self.assertIn("--mini_test 5", script)
        self.assertIn("--mini_test 2", script)
        self.assertIn("--method base", script)

    def test_build_jobs_supports_beam_search_for_new_model(self):
        jobs = build_jobs(
            model="llava-v1.5-7b",
            pope_split="random",
            pope_count=3,
            chair_count=0,
            methods=["beam_search"],
            datasets=["pope"],
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].name, "pope_beam_search")
        self.assertIn("--model", jobs[0].args)
        self.assertIn("llava-v1.5-7b", jobs[0].args)
        self.assertIn("beam_search", jobs[0].args)


if __name__ == "__main__":
    unittest.main()
