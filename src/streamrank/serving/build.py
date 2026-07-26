from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from streamrank.ranking.scoring import PolicyScorer
from streamrank.serving.manifest import DeploymentBundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_descriptor(
    path: Path,
    component: str,
    version: str,
    compatibility_key: str,
    config: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "component": component,
                "version": version,
                "compatibility_key": compatibility_key,
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_serving_deployment(
    root: str | Path,
    *,
    catalog_path: str | Path,
    policy_path: str | Path = "configs/serving_policy.json",
    focused_artifact_path: str | Path | None = None,
    manifest_name: str = "kuairand-pure-sample",
) -> Path:
    """Bind catalog, offline-selected ranker and an explicit rerank policy into one manifest.

    Score weights are read from a declared policy config instead of a tuning report: this
    prototype pre-registers its product weights and does not claim a logged-data policy search.
    """
    root = Path(root).resolve()
    catalog_path = Path(catalog_path).resolve()
    policy_path = Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    # resolve unconditionally: on macOS /tmp is a symlink to /private/tmp, and an
    # unresolved absolute path would fail relative_to() against the resolved root.
    policy_path = policy_path.resolve()
    if not catalog_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("catalog and serving policy config must exist")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    weights = dict(policy["score_weights"])
    PolicyScorer(weights)  # fail fast on sign conventions before writing any descriptor
    dataset_manifest_path = catalog_path.parent / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.is_file()
        else {}
    )
    dataset_name = str(dataset_manifest.get("dataset", "KuaiRand-Pure"))
    dataset_mode = str(dataset_manifest.get("mode", "unknown cohort"))
    selected_users = dataset_manifest.get("selected_users", "unknown")
    compatibility_key = "kuairand-pure-sample-schema-v1"
    components_dir = root / "deployments/components"
    manifests_dir = root / "deployments/manifests"
    components_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    focused_artifact_path = (
        Path(focused_artifact_path).resolve()
        if focused_artifact_path is not None
        else root / "artifacts/sequence-ranking-real/artifact.json"
    )
    if not focused_artifact_path.is_file():
        raise FileNotFoundError("focused serving artifact must exist")
    focused_artifact = json.loads(focused_artifact_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(focused_artifact["checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("focused serving checkpoint must exist")
    winner = str(focused_artifact["winner"])
    model_spec = focused_artifact["model"]

    versions = {
        "model": f"focused-{winner}-kuairand-pure-v1",
        "calibrator": "identity-kuairand-pure-v1",
        "feature_schema": "pit-kuairand-pure-v1",
        "item_index": "itemcf-popularity-kuairand-pure-v1",
        "rerank_policy": "policy-kuairand-pure-v1",
    }
    paths = {
        name: components_dir / f"{name.replace('_', '-')}-{version}.json"
        for name, version in versions.items()
    }
    _write_descriptor(
        paths["model"],
        "model",
        versions["model"],
        compatibility_key,
        {
            "kind": "focused_ranker",
            "architecture": winner,
            "task_layer": model_spec.get("task_layer"),
            "artifact_path": str(Path("../../") / focused_artifact_path.relative_to(root)),
            "artifact_sha256": _sha256(focused_artifact_path),
            "checkpoint_path": str(Path("../../") / checkpoint_path.relative_to(root)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_note": "offline-selected focused ranker bound to online serving rerank stage",
        },
    )
    _write_descriptor(
        paths["calibrator"],
        "calibrator",
        versions["calibrator"],
        compatibility_key,
        {
            "kind": "identity",
            "tasks": ["is_click", "long_view", "is_like", "is_hate"],
            "serving_note": "offline Platt calibrators are not claimed as bound serving artifacts",
        },
    )
    _write_descriptor(
        paths["feature_schema"],
        "feature_schema",
        versions["feature_schema"],
        compatibility_key,
        {
            "required_candidate_features": ["category", "author_id", "upload_time_ms"],
            "probability_outputs": ["is_click", "long_view", "is_like", "is_hate"],
            "source": "KuaiRand-Pure video_features_basic_pure.csv",
        },
    )
    relative_catalog = Path("../../") / catalog_path.relative_to(root)
    _write_descriptor(
        paths["item_index"],
        "item_index",
        versions["item_index"],
        compatibility_key,
        {
            "kind": "itemcf_popularity",
            "catalog_path": str(relative_catalog),
            "catalog_sha256": _sha256(catalog_path),
            "retrievers": ["itemcf", "popularity"],
            "dataset": f"{dataset_name} {dataset_mode}; {selected_users} users",
            "dataset_manifest": (
                str(dataset_manifest_path.relative_to(root))
                if dataset_manifest_path.is_file()
                else None
            ),
            "fit_scope": (
                "full prepared catalog log for interactive serving; separate from "
                "offline temporal evaluation"
            ),
        },
    )
    _write_descriptor(
        paths["rerank_policy"],
        "rerank_policy",
        versions["rerank_policy"],
        compatibility_key,
        {
            "score_weights": weights,
            "max_per_category": int(policy.get("max_per_category", 5)),
            "max_per_author": int(policy.get("max_per_author", 3)),
            "concentration_penalty": float(policy.get("concentration_penalty", 0.05)),
            "weight_source": str(policy_path.relative_to(root)),
            "weight_selection": policy.get("weight_selection", {"method": "pre_registered"}),
        },
    )

    manifest_path = manifests_dir / f"{manifest_name}.json"
    manifest = {
        "deployment_id": f"{manifest_name}-v1",
        "model_version": versions["model"],
        "calibrator_version": versions["calibrator"],
        "feature_schema_version": versions["feature_schema"],
        "item_index_version": versions["item_index"],
        "rerank_policy_version": versions["rerank_policy"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "compatibility_key": compatibility_key,
        "component_compatibility": {name: compatibility_key for name in versions},
        "artifacts": {
            name: {
                "path": f"../components/{path.name}",
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    DeploymentBundle.load(manifest_path)
    return manifest_path
