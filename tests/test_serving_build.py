import json
import tempfile
import unittest
from pathlib import Path

from streamrank.serving.build import build_serving_deployment
from streamrank.serving.manifest import DeploymentBundle


class ServingBuildTests(unittest.TestCase):
    def _write_fixtures(self, root: Path) -> tuple[Path, Path]:
        catalog = root / "data/processed/sample/interactions.csv"
        policy = root / "configs/serving_policy.json"
        focused = root / "artifacts/sequence-ranking-real/artifact.json"
        checkpoint = root / "artifacts/sequence-ranking-real/deepfm.pt"
        catalog.parent.mkdir(parents=True)
        policy.parent.mkdir(parents=True)
        focused.parent.mkdir(parents=True)
        (catalog.parent / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "KuaiRand-Pure",
                    "mode": "deterministic-user-sample",
                    "selected_users": 7,
                }
            ),
            encoding="utf-8",
        )
        catalog.write_text(
            "user_id,item_id,event_time_ms,category,author_id,upload_time_ms\n1,2,3,tag:1,4,1\n",
            encoding="utf-8",
        )
        policy.write_text(
            json.dumps(
                {
                    "score_weights": {
                        "is_click": 0.4,
                        "long_view": 0.3,
                        "is_like": 0.2,
                        "is_hate": -0.8,
                        "freshness": 0.1,
                    },
                    "max_per_category": 5,
                    "max_per_author": 3,
                    "concentration_penalty": 0.05,
                    "weight_selection": {"method": "pre_registered"},
                }
            ),
            encoding="utf-8",
        )
        checkpoint.write_bytes(b"placeholder checkpoint")
        focused.write_text(
            json.dumps(
                {
                    "winner": "deepfm",
                    "model": {
                        "architecture": "deepfm",
                        "task_layer": "shared_bottom",
                        "embedding_dim": 32,
                        "hidden_dim": 64,
                    },
                    "checkpoint": "artifacts/sequence-ranking-real/deepfm.pt",
                }
            ),
            encoding="utf-8",
        )
        return catalog, policy

    def test_builds_a_complete_validated_real_catalog_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, policy = self._write_fixtures(root)
            manifest_path = build_serving_deployment(root, catalog_path=catalog, policy_path=policy)
            bundle = DeploymentBundle.load(manifest_path)
            self.assertEqual(
                bundle.components["item_index"]["dataset"],
                "KuaiRand-Pure deterministic-user-sample; 7 users",
            )
            self.assertEqual(bundle.manifest.deployment_id, "kuairand-pure-sample-v1")
            self.assertEqual(bundle.components["model"]["kind"], "focused_ranker")
            self.assertEqual(bundle.components["model"]["architecture"], "deepfm")
            rerank = bundle.components["rerank_policy"]
            self.assertEqual(rerank["weight_source"], "configs/serving_policy.json")
            self.assertEqual(rerank["weight_selection"], {"method": "pre_registered"})
            self.assertEqual(rerank["score_weights"]["is_hate"], -0.8)

    def test_rejects_policy_with_wrong_sign_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, policy = self._write_fixtures(root)
            broken = json.loads(policy.read_text(encoding="utf-8"))
            broken["score_weights"]["is_hate"] = 0.5
            policy.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-positive weight"):
                build_serving_deployment(root, catalog_path=catalog, policy_path=policy)


if __name__ == "__main__":
    unittest.main()
