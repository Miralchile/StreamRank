import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from streamrank.serving.manifest import DeploymentManifest, ManifestStore


class ManifestTests(unittest.TestCase):
    def test_atomic_manifest_activation_switches_all_versions(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "deployments/manifests/demo.json"
        source = DeploymentManifest.load(source_path)
        artifacts = {
            name: {
                **reference,
                "path": str((source_path.parent / reference["path"]).resolve()),
            }
            for name, reference in source.artifacts.items()
        }
        manifest = replace(source, deployment_id="d1", artifacts=artifacts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            ManifestStore(path).activate(manifest)
            loaded = ManifestStore(path).current()
            self.assertEqual(loaded, manifest)
            payload = json.loads(path.read_text())
            self.assertEqual(set(payload), set(manifest.__dict__))

    def test_rejects_missing_or_tampered_artifact(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "deployments/manifests/demo.json").read_text())
        payload["artifacts"]["model"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for reference in payload["artifacts"].values():
                reference["path"] = str(
                    (root / "deployments/manifests" / reference["path"]).resolve()
                )
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                DeploymentManifest.load(path)

    def test_failed_activation_preserves_previous_complete_manifest(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "deployments/manifests/demo.json"
        source = DeploymentManifest.load(source_path)
        absolute_artifacts = {
            name: {**ref, "path": str((source_path.parent / ref["path"]).resolve())}
            for name, ref in source.artifacts.items()
        }
        valid = replace(source, artifacts=absolute_artifacts)
        invalid_artifacts = {
            name: dict(reference) for name, reference in absolute_artifacts.items()
        }
        invalid_artifacts["model"]["sha256"] = "0" * 64
        invalid = replace(valid, deployment_id="invalid", artifacts=invalid_artifacts)
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory) / "active.json")
            store.activate(valid)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                store.activate(invalid)
            self.assertEqual(store.current().deployment_id, valid.deployment_id)

    def test_catalog_digest_failure_preserves_active_manifest(self):
        root = Path(__file__).resolve().parents[1]
        source_path = root / "deployments/manifests/demo.json"
        source = DeploymentManifest.load(source_path)
        absolute_artifacts = {
            name: {**ref, "path": str((source_path.parent / ref["path"]).resolve())}
            for name, ref in source.artifacts.items()
        }
        valid = replace(source, artifacts=absolute_artifacts)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            descriptor = json.loads(
                (root / "deployments/components/item-index-runtime-v1.json").read_text()
            )
            descriptor["config"]["catalog_path"] = str(root / "data/demo/interactions.csv")
            descriptor["config"]["catalog_sha256"] = "0" * 64
            bad_descriptor = directory_path / "bad-index.json"
            bad_descriptor.write_text(json.dumps(descriptor))
            bad_artifacts = {name: dict(ref) for name, ref in absolute_artifacts.items()}
            bad_artifacts["item_index"] = {
                "path": str(bad_descriptor),
                "sha256": hashlib.sha256(bad_descriptor.read_bytes()).hexdigest(),
            }
            invalid = replace(valid, deployment_id="bad-catalog", artifacts=bad_artifacts)
            store = ManifestStore(directory_path / "active.json")
            store.activate(valid)
            with self.assertRaisesRegex(ValueError, "catalog digest mismatch"):
                store.activate(invalid)
            self.assertEqual(store.current().deployment_id, valid.deployment_id)

    def test_rejects_component_compatibility_mismatch(self):
        manifest = DeploymentManifest(
            "d1",
            "m1",
            "c1",
            "f1",
            "i1",
            "r1",
            "now",
            "schema1",
            {"model": "schema2"},
        )
        with self.assertRaises(ValueError):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()
