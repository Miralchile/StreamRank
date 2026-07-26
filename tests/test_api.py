import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from streamrank.serving.app import create_app

    FASTAPI_AVAILABLE = True
except (ImportError, RuntimeError):
    # starlette raises RuntimeError (not ImportError) when its test client
    # dependency (httpx2) is missing; both cases mean "skip the API suite".
    FASTAPI_AVAILABLE = False


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE and TORCH_AVAILABLE, "serving and ml extras are required")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_health_and_recommendation(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertGreater(health.json()["fit_cutoff_ms"], 0)
        self.assertIn("serving_fit_scope", health.json())
        response = self.client.get(
            "/recommend/1",
            params={"top_k": 3, "query_time_ms": health.json()["fit_cutoff_ms"] + 1},
        )
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(response.json()["items"]), 3)
        self.assertIn("deployment_id", response.json())
        self.assertIn("ranker", response.json())
        pipeline = response.json()["pipeline"]
        stages = {entry["stage"]: entry["count"] for entry in pipeline["stages"]}
        self.assertEqual(
            list(stages),
            ["recall_itemcf", "recall_popularity", "fusion", "ranking", "rerank"],
        )
        self.assertEqual(stages["rerank"], len(response.json()["items"]))
        self.assertGreaterEqual(stages["fusion"], stages["rerank"])
        self.assertGreater(pipeline["catalog_size"], 0)

    def test_dashboard_and_curated_project_endpoints(self):
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("StreamRank", dashboard.text)
        self.assertIn("真实数据", dashboard.text)
        self.assertIn("训练与选型", dashboard.text)
        self.assertIn("效果验证", dashboard.text)
        self.assertIn('name="top_k" type="number" min="1" max="20" step="1"', dashboard.text)
        self.assertNotIn('name="top_k" type="range"', dashboard.text)
        self.assertIn('name="cohort_size" type="number"', dashboard.text)
        self.assertIn('name="history_length" type="number"', dashboard.text)
        self.assertIn("模拟长播反馈并刷新", dashboard.text)
        self.assertIn('id="pipelineFunnel"', dashboard.text)
        stylesheet = self.client.get("/assets/styles.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("--acid", stylesheet.text)
        self.assertIn(".funnel-stage", stylesheet.text)

        # The public API surface stays minimal: retired curated endpoints must not resurface.
        for retired in ("/api/project", "/api/dataset", "/api/experiment", "/api/benchmark"):
            self.assertEqual(self.client.get(retired).status_code, 404, retired)

        focused = self.client.get("/api/focused")
        self.assertEqual(focused.status_code, 200)
        self.assertEqual(focused.json()["experiment"]["winner"], "autoint")
        self.assertEqual(focused.json()["dataset"]["selected_users"], 5000)
        self.assertTrue(focused.json()["serving"]["offline_winner_bound"])

        profiles = self.client.get("/api/serving-users")
        self.assertEqual(profiles.status_code, 200)
        self.assertGreater(len(profiles.json()["profiles"]), 0)
        self.assertGreater(profiles.json()["profiles"][0]["history_size"], 0)

    def test_event_ingestion_is_idempotent(self):
        payload = {
            "user_id": 1,
            "item_id": 108,
            "event_time_ms": 1650594000000,
            "long_view": 1,
            "category": "travel",
        }
        first = self.client.post("/events", json=payload, headers={"X-Event-ID": "api-test-1"})
        second = self.client.post("/events", json=payload, headers={"X-Event-ID": "api-test-1"})
        self.assertTrue(first.json()["applied"])
        self.assertTrue(second.json()["deduplicated"])

    def test_metrics_are_prometheus_text(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("streamrank_requests_total", response.text)
        self.assertIn("streamrank_latency_p95_ms", response.text)

    def test_ready_and_binary_label_validation(self):
        self.assertEqual(self.client.get("/ready").status_code, 200)
        response = self.client.post(
            "/events",
            json={"user_id": 1, "item_id": 1, "event_time_ms": 1, "is_click": 2},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.client.get("/recommend/1", params={"query_time_ms": 1}).status_code,
            400,
        )

    def test_kafka_event_mode_publishes_instead_of_direct_state_write(self):
        root = Path(__file__).resolve().parents[1]
        with (
            patch.dict(
                os.environ,
                {
                    "STREAMRANK_ROOT": str(root),
                    "STREAMRANK_STATE_BACKEND": "memory",
                    "STREAMRANK_EVENT_MODE": "kafka",
                },
            ),
            patch("streamrank.serving.app.KafkaEventProducer") as producer_type,
        ):
            kafka_client = TestClient(create_app())
            response = kafka_client.post(
                "/events",
                json={"user_id": 1, "item_id": 2, "event_time_ms": 3},
                headers={"X-Event-ID": "kafka-api-test"},
            )
        self.assertEqual(
            response.json(),
            {"accepted": True, "applied": True, "deduplicated": False, "mode": "kafka+sync"},
        )
        producer_type.return_value.send.assert_called_once()

    def test_runtime_root_can_be_explicitly_configured(self):
        from streamrank.serving.app import _resolve_runtime_root

        with patch.dict(os.environ, {"STREAMRANK_ROOT": os.getcwd()}):
            self.assertEqual(_resolve_runtime_root(), Path.cwd().resolve())

    def test_real_profile_rejects_mismatched_benchmark(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / "benchmark.json"
            benchmark.write_text(
                json.dumps({"deployment_id": "wrong-deployment"}), encoding="utf-8"
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "STREAMRANK_ROOT": str(root),
                        "STREAMRANK_MANIFEST": str(
                            root / "deployments/manifests/kuairand-pure-sample.json"
                        ),
                        "STREAMRANK_BENCHMARK_REPORT": str(benchmark),
                        "STREAMRANK_STATE_BACKEND": "memory",
                        "STREAMRANK_EVENT_MODE": "sync",
                    },
                    clear=False,
                ),
                self.assertRaisesRegex(ValueError, "benchmark report deployment_id"),
            ):
                create_app()

    def test_invalid_primary_manifest_uses_complete_fallback_bundle(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict(
            os.environ,
            {
                "STREAMRANK_MANIFEST": str(root / "missing-manifest.json"),
                "STREAMRANK_FALLBACK_MANIFEST": str(root / "deployments/manifests/fallback.json"),
                "STREAMRANK_STATE_BACKEND": "memory",
                "STREAMRANK_EVENT_MODE": "sync",
            },
        ):
            fallback_client = TestClient(create_app())
        health = fallback_client.get("/health").json()
        self.assertEqual(health["deployment_id"], "fallback-2026-07-17")
        self.assertEqual(health["degraded_reason"], "primary_manifest_load_failed")

    def test_serving_incompatible_primary_uses_fallback_bundle(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "deployments/manifests/demo.json"
        payload = json.loads(source_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model = json.loads(
                (root / "deployments/components/model-heuristic-v1.json").read_text()
            )
            model["config"]["kind"] = "unsupported-neural-ranker"
            bad_model = directory_path / "bad-model.json"
            bad_model.write_text(json.dumps(model))
            for reference in payload["artifacts"].values():
                reference["path"] = str((source_path.parent / reference["path"]).resolve())
            payload["artifacts"]["model"] = {
                "path": str(bad_model),
                "sha256": hashlib.sha256(bad_model.read_bytes()).hexdigest(),
            }
            primary = directory_path / "primary.json"
            primary.write_text(json.dumps(payload))
            with patch.dict(
                os.environ,
                {
                    "STREAMRANK_MANIFEST": str(primary),
                    "STREAMRANK_FALLBACK_MANIFEST": str(
                        root / "deployments/manifests/fallback.json"
                    ),
                    "STREAMRANK_STATE_BACKEND": "memory",
                    "STREAMRANK_EVENT_MODE": "sync",
                },
            ):
                fallback_client = TestClient(create_app())
        self.assertEqual(
            fallback_client.get("/health").json()["deployment_id"],
            "fallback-2026-07-17",
        )


if __name__ == "__main__":
    unittest.main()
