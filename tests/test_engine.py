import unittest
from pathlib import Path

from streamrank.data.loader import load_interactions
from streamrank.domain import Interaction
from streamrank.engine import RecommendationEngine
from streamrank.ranking.scoring import PolicyScorer
from streamrank.serving.manifest import DeploymentManifest


class EngineTests(unittest.TestCase):
    def test_end_to_end_recommendation_has_provenance_and_score(self):
        root = Path(__file__).resolve().parents[1]
        events = list(load_interactions(root / "data/demo/interactions.csv"))
        manifest = DeploymentManifest("d", "m", "c", "f", "i", "r", "now", "schema")
        scorer = PolicyScorer(
            {
                "is_click": 0.4,
                "long_view": 0.3,
                "is_like": 0.2,
                "freshness": 0.1,
                "is_hate": -0.8,
            }
        )
        engine = RecommendationEngine(manifest, scorer).fit(events)
        trace: dict[str, object] = {}
        rows = engine.recommend(
            1, top_k=3, query_time_ms=max(row.event_time_ms for row in events) + 1, trace=trace
        )
        self.assertTrue(rows)
        self.assertTrue(rows[0].retrieval_sources)
        self.assertIsInstance(rows[0].rank_score, float)
        self.assertEqual(trace["returned"], len(rows))
        self.assertGreaterEqual(trace["fused"], trace["returned"])
        self.assertGreaterEqual(trace["recall_itemcf"] + trace["recall_popularity"], trace["fused"])
        self.assertGreater(trace["catalog_size"], 0)

    def test_rejects_query_before_fitted_artifact_cutoff(self):
        events = [
            Interaction(1, 10, 1000, long_view=1),
            Interaction(1, 11, 3000, long_view=1),
        ]
        manifest = DeploymentManifest("d", "m", "c", "f", "i", "r", "now", "schema")
        engine = RecommendationEngine(manifest, PolicyScorer({"long_view": 1.0})).fit(events)
        with self.assertRaisesRegex(ValueError, "precedes the fitted artifact cutoff"):
            engine.recommend(1, query_time_ms=2000)
        self.assertEqual(engine.state.history(1, before_ms=2000), [10])


if __name__ == "__main__":
    unittest.main()
