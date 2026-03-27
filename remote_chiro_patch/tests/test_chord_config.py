import argparse
import unittest

from chord.config import CHORDConfig


class CHORDConfigTests(unittest.TestCase):
    def test_defaults_do_not_enable_chord_without_explicit_args(self) -> None:
        args = argparse.Namespace(
            chord_enable=False,
            alpha_anchor=None,
            lambda_cur=None,
            lambda_fut=None,
            lambda_txt=None,
            future_horizon=None,
            future_topk=None,
            detector_box_threshold=None,
            detector_text_threshold=None,
            max_boxes=None,
            grounding_dino_path=None,
            detector_python=None,
            chord_log_jsonl=None,
        )

        cfg = CHORDConfig.from_args(args)

        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.lambda_cur, 0.25)
        self.assertEqual(cfg.lambda_fut, 0.5)

    def test_explicit_lambda_enables_chord_without_flag(self) -> None:
        args = argparse.Namespace(
            chord_enable=False,
            alpha_anchor=None,
            lambda_cur=0.1,
            lambda_fut=None,
            lambda_txt=None,
            future_horizon=None,
            future_topk=None,
            detector_box_threshold=None,
            detector_text_threshold=None,
            max_boxes=None,
            grounding_dino_path=None,
            detector_python=None,
            chord_log_jsonl=None,
        )

        cfg = CHORDConfig.from_args(args)

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.lambda_cur, 0.1)


if __name__ == "__main__":
    unittest.main()
