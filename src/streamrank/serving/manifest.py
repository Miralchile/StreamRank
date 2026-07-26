from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

COMPONENT_VERSION_FIELDS = {
    "model": "model_version",
    "calibrator": "calibrator_version",
    "feature_schema": "feature_schema_version",
    "item_index": "item_index_version",
    "rerank_policy": "rerank_policy_version",
}


@dataclass(frozen=True)
class DeploymentManifest:
    deployment_id: str
    model_version: str
    calibrator_version: str
    feature_schema_version: str
    item_index_version: str
    rerank_policy_version: str
    created_at: str
    compatibility_key: str
    component_compatibility: dict[str, str] | None = None
    artifacts: dict[str, dict[str, str]] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentManifest":
        manifest_path = Path(path).resolve()
        manifest = cls(**json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest.validate()
        manifest.load_artifact_configs(manifest_path)
        return manifest

    def validate(self) -> None:
        required_versions = {
            "model": self.model_version,
            "calibrator": self.calibrator_version,
            "feature_schema": self.feature_schema_version,
            "item_index": self.item_index_version,
            "rerank_policy": self.rerank_policy_version,
        }
        empty = [name for name, version in required_versions.items() if not version.strip()]
        if empty:
            raise ValueError(f"deployment manifest has empty component versions: {empty}")
        if not self.compatibility_key.strip():
            raise ValueError("deployment manifest compatibility_key is empty")
        if self.component_compatibility:
            mismatched = {
                name: key
                for name, key in self.component_compatibility.items()
                if key != self.compatibility_key
            }
            if mismatched:
                message = (
                    f"component compatibility mismatch; expected "
                    f"{self.compatibility_key}: {mismatched}"
                )
                raise ValueError(message)

    def load_artifact_configs(self, manifest_path: str | Path) -> dict[str, dict[str, object]]:
        """Verify immutable component descriptors before a deployment can be served."""
        if not self.artifacts:
            raise ValueError("deployment manifest must bind immutable component artifacts")
        missing = sorted(set(COMPONENT_VERSION_FIELDS) - set(self.artifacts))
        if missing:
            raise ValueError(f"deployment manifest is missing artifacts: {missing}")
        base = Path(manifest_path).resolve().parent
        configs: dict[str, dict[str, object]] = {}
        for component, version_field in COMPONENT_VERSION_FIELDS.items():
            reference = self.artifacts[component]
            artifact_path = Path(reference.get("path", ""))
            if not artifact_path.is_absolute():
                artifact_path = (base / artifact_path).resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(f"missing {component} artifact: {artifact_path}")
            content = artifact_path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != reference.get("sha256"):
                raise ValueError(f"{component} artifact digest mismatch")
            descriptor = json.loads(content)
            if descriptor.get("component") != component:
                raise ValueError(f"{component} descriptor has the wrong component name")
            if descriptor.get("version") != getattr(self, version_field):
                raise ValueError(f"{component} descriptor version mismatch")
            if descriptor.get("compatibility_key") != self.compatibility_key:
                raise ValueError(f"{component} descriptor compatibility mismatch")
            config = descriptor.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"{component} descriptor config must be an object")
            configs[component] = {**config, "_artifact_dir": str(artifact_path.parent)}
        return configs


@dataclass(frozen=True)
class DeploymentBundle:
    manifest: DeploymentManifest
    components: dict[str, dict[str, object]]
    manifest_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "DeploymentBundle":
        manifest_path = Path(path).resolve()
        manifest = DeploymentManifest.load(manifest_path)
        components = manifest.load_artifact_configs(manifest_path)
        index = components["item_index"]
        catalog_path = Path(str(index.get("catalog_path", "")))
        if not catalog_path.is_absolute():
            catalog_path = (Path(str(index["_artifact_dir"])) / catalog_path).resolve()
        if not catalog_path.is_file():
            raise FileNotFoundError(f"missing bound catalog: {catalog_path}")
        catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        if catalog_digest != index.get("catalog_sha256"):
            raise ValueError("bound catalog digest mismatch")
        index["catalog_path"] = str(catalog_path)
        return cls(manifest, components, manifest_path)


class ManifestStore:
    """Atomically activates all coupled serving components as one deployment."""

    def __init__(self, active_manifest_path: str | Path):
        self.active_manifest_path = Path(active_manifest_path)

    def current(self) -> DeploymentManifest:
        return DeploymentBundle.load(self.active_manifest_path).manifest

    def activate(self, manifest: DeploymentManifest) -> None:
        manifest.validate()
        self.active_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".manifest-", suffix=".json", dir=self.active_manifest_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(manifest), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            # The temporary file lives beside the final manifest, so relative artifact
            # references resolve exactly as they will after activation.
            DeploymentBundle.load(temporary)
            os.replace(temporary, self.active_manifest_path)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
