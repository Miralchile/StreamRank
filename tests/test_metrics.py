import unittest

from streamrank.evaluation.metrics import binary_classification_report, gauc, pr_auc, roc_auc


class MetricTests(unittest.TestCase):
    def test_auc_handles_ties(self):
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(roc_auc([0, 1], [0.1, 0.9]), 1.0)

    def test_binary_report(self):
        report = binary_classification_report([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7])
        self.assertGreater(report["roc_auc"], 0.9)
        self.assertIn("pr_auc", report)
        self.assertIn("ece_10", report)

    def test_pr_auc_is_invariant_to_order_with_tied_scores(self):
        first = pr_auc([1, 0, 1], [0.5, 0.5, 0.2])
        second = pr_auc([0, 1, 1], [0.5, 0.5, 0.2])
        self.assertAlmostEqual(first, second)

    def test_gauc_reports_valid_group_fraction(self):
        report = gauc([0, 1, 1, 1], [0.1, 0.9, 0.8, 0.7], ["u1", "u1", "u2", "u2"])
        self.assertEqual(report["valid_groups"], 1)
        self.assertEqual(report["valid_group_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
