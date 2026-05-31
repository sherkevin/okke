from __future__ import annotations

import unittest


class PopePredictionAccountingTest(unittest.TestCase):
    def test_unknown_prediction_is_not_treated_as_true_negative(self):
        tp = fp = fn = tn = 0
        label = "no"
        pred = "unknown"

        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "yes" and pred == "no":
            fn += 1
        elif label == "no" and pred == "no":
            tn += 1

        self.assertEqual((tp, fp, fn, tn), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
