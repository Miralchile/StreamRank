import unittest

from streamrank.data.audit import audit_interactions
from streamrank.domain import Interaction


class AuditTests(unittest.TestCase):
    def test_reports_label_implication_violations_and_missing_request_ids(self):
        report = audit_interactions(
            [
                Interaction(1, 1, 1000, is_click=0, long_view=1),
                Interaction(1, 2, 2000, is_click=1, long_view=1),
            ]
        )
        self.assertEqual(report.long_view_without_click, 1)
        self.assertEqual(report.request_id_coverage, 0)
        self.assertTrue(any("request_id" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
