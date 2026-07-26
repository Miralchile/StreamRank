import unittest

from streamrank.domain import Interaction
from streamrank.evaluation.recall import build_queries, evaluate_recall


def _events():
    rows = []
    # Users 1-3 co-watch items 10 and 11 before the cutoff (long views), which
    # gives ItemCF a strong 10<->11 association. User 4 has watched 10 only.
    for user in (1, 2, 3):
        rows.append(Interaction(user, 10, 1_000 + user, long_view=1, category="a"))
        rows.append(Interaction(user, 11, 2_000 + user, long_view=1, category="a"))
    rows.append(Interaction(4, 10, 1_500, long_view=1, category="a"))
    # Popularity noise from another category before the cutoff.
    rows.append(Interaction(5, 20, 1_600, long_view=1, category="b"))
    # After the cutoff user 4 long-views item 11 (fresh target) and item 10
    # again (already seen -> must not count as a target).
    rows.append(Interaction(4, 11, 6_000, long_view=1, category="a"))
    rows.append(Interaction(4, 10, 6_100, long_view=1, category="a"))
    return rows


class RecallEvalTests(unittest.TestCase):
    def test_targets_exclude_previously_seen_items(self):
        queries = build_queries(_events(), fit_end_ms=5_000, eval_end_ms=10_000)
        self.assertEqual(len(queries), 1)
        query = queries[0]
        self.assertEqual(query.user_id, 4)
        self.assertEqual(query.targets, {11})
        self.assertEqual(query.history, [10])
        self.assertIn(10, query.seen)

    def test_itemcf_and_fusion_recover_the_fresh_target(self):
        report = evaluate_recall(_events(), fit_end_ms=5_000, eval_end_ms=10_000, ks=(1, 5))
        self.assertEqual(report["evaluated_users"], 1)
        self.assertEqual(report["sources"]["itemcf"]["@1"]["recall"], 1.0)
        self.assertEqual(report["sources"]["rrf_fusion"]["@5"]["hit_rate"], 1.0)
        self.assertLessEqual(report["sources"]["popularity"]["@1"]["recall"], 1.0)
        self.assertGreater(report["sources"]["itemcf"]["catalog_coverage"], 0.0)

    def test_rejects_windows_without_evaluable_users(self):
        with self.assertRaises(ValueError):
            evaluate_recall(_events(), fit_end_ms=5_000, eval_end_ms=5_001)


if __name__ == "__main__":
    unittest.main()
