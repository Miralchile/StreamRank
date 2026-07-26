import unittest

from streamrank.ranking.calibration import PlattCalibrator
from streamrank.ranking.scoring import PolicyScorer


class PolicyScorerTests(unittest.TestCase):
    def setUp(self):
        self.scorer = PolicyScorer(
            {
                "is_click": 0.4,
                "long_view": 0.3,
                "is_like": 0.2,
                "freshness": 0.1,
                "is_hate": -0.8,
            }
        )

    def test_positive_objective_increases_descending_score(self):
        base = {"is_click": 0.2, "long_view": 0.2, "is_like": 0.1, "is_hate": 0.01}
        improved = dict(base, long_view=0.8)
        self.assertGreater(self.scorer.score(improved), self.scorer.score(base))

    def test_hate_decreases_descending_score(self):
        base = {"is_click": 0.5, "long_view": 0.4, "is_like": 0.1, "is_hate": 0.01}
        harmful = dict(base, is_hate=0.5)
        self.assertLess(self.scorer.score(harmful), self.scorer.score(base))

    def test_rejects_sign_errors(self):
        with self.assertRaises(ValueError):
            PolicyScorer({"long_view": -0.3})
        with self.assertRaises(ValueError):
            PolicyScorer({"is_hate": 0.3})

    def test_platt_calibration_does_not_reverse_ordering(self):
        calibrator = PlattCalibrator().fit([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
        self.assertGreaterEqual(calibrator.slope, 0)
        self.assertLessEqual(calibrator.predict(0.1), calibrator.predict(0.9))


if __name__ == "__main__":
    unittest.main()
