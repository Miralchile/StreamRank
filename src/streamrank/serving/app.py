from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install StreamRank with the 'serving' extra") from exc

from streamrank.data.loader import load_interactions
from streamrank.domain import Interaction
from streamrank.engine import RecommendationEngine
from streamrank.online.state import InMemoryOnlineState, RedisOnlineState
from streamrank.ranking.scoring import PolicyScorer
from streamrank.serving.manifest import DeploymentBundle, DeploymentManifest
from streamrank.streaming.producer import KafkaEventProducer


class EventPayload(BaseModel):
    user_id: int
    item_id: int
    event_time_ms: int
    is_click: int = Field(default=0, ge=0, le=1)
    long_view: int = Field(default=0, ge=0, le=1)
    is_like: int = Field(default=0, ge=0, le=1)
    is_hate: int = Field(default=0, ge=0, le=1)
    tab: int = -1
    is_rand: int = Field(default=0, ge=0, le=1)
    logging_policy: str = "api"
    request_id: str | None = None
    category: str = "UNKNOWN"
    author_id: int = -1
    upload_time_ms: int = 0


def _resolve_runtime_root() -> Path:
    """Resolve runtime assets without assuming an editable source installation."""
    configured = os.getenv("STREAMRANK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd().resolve()


def _load_serving_bundle(path: str | Path) -> tuple[DeploymentBundle, PolicyScorer]:
    """Load every artifact and enforce binary-specific compatibility as one operation."""
    bundle = DeploymentBundle.load(path)
    components = bundle.components
    if components["model"].get("kind") not in {"heuristic", "focused_ranker"}:
        raise ValueError("this serving binary only supports heuristic or focused ranker models")
    if components["calibrator"].get("kind") != "identity":
        raise ValueError("this serving binary only supports the bound identity calibrator")
    if components["item_index"].get("kind") != "itemcf_popularity":
        raise ValueError("unsupported bound item index kind")
    required_features = set(components["feature_schema"].get("required_candidate_features", []))
    if required_features != {"category", "author_id", "upload_time_ms"}:
        raise ValueError("bound feature schema is incompatible with this serving binary")
    scorer = PolicyScorer(components["rerank_policy"]["score_weights"])
    return bundle, scorer


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"invalid runtime artifact: {path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"invalid runtime artifact: {path.name}")
    return payload


def _load_json_file(path: Path) -> dict[str, object]:
    """Load a startup artifact and fail closed before serving mixed profiles."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"runtime artifact must be an object: {path}")
    return payload


def create_app() -> FastAPI:
    root = _resolve_runtime_root()
    default_manifest_path = root / "deployments/manifests/kuairand-pure-sample.json"
    if not default_manifest_path.is_file():
        default_manifest_path = root / "deployments/manifests/demo.json"
    manifest_path = Path(os.getenv("STREAMRANK_MANIFEST", default_manifest_path))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    fallback_reason = ""
    try:
        bundle, scorer = _load_serving_bundle(manifest_path)
    except (FileNotFoundError, ValueError):
        fallback_path = os.getenv("STREAMRANK_FALLBACK_MANIFEST")
        if not fallback_path:
            raise
        fallback = Path(fallback_path)
        if not fallback.is_absolute():
            fallback = root / fallback
        bundle, scorer = _load_serving_bundle(fallback)
        fallback_reason = "primary_manifest_load_failed"
    manifest = bundle.manifest
    model_config = bundle.components["model"]
    index_config = bundle.components["item_index"]
    rerank_config = bundle.components["rerank_policy"]
    is_real_profile = "kuairand-pure" in str(index_config.get("dataset", "")).lower()
    state_backend = os.getenv("STREAMRANK_STATE_BACKEND", "memory").lower()
    if state_backend == "redis":
        state = RedisOnlineState(os.getenv("STREAMRANK_REDIS_URL", "redis://localhost:6379/0"))
        state.ping()
    elif state_backend == "memory":
        state = InMemoryOnlineState()
    else:
        raise ValueError(f"unsupported STREAMRANK_STATE_BACKEND={state_backend}")
    engine = RecommendationEngine(
        manifest,
        scorer,
        state=state,
        model_config=model_config,
        rerank_config=rerank_config,
    )
    catalog_path = Path(str(index_config["catalog_path"]))
    engine.fit(load_interactions(catalog_path))
    if fallback_reason:
        engine.degraded_count += 1
        engine.last_degraded_reason = fallback_reason

    event_mode = os.getenv("STREAMRANK_EVENT_MODE", "sync").lower()
    producer = None
    if event_mode == "kafka":
        producer = KafkaEventProducer(
            os.getenv("STREAMRANK_KAFKA_BOOTSTRAP", "localhost:19092"),
            os.getenv("STREAMRANK_EVENT_TOPIC", "streamrank.events"),
        )
    elif event_mode != "sync":
        raise ValueError(f"unsupported STREAMRANK_EVENT_MODE={event_mode}")

    app = FastAPI(title="StreamRank", version="0.1.0")
    app.state.engine = engine
    default_dataset_manifest = (
        root / "data/processed/kuairand-pure-5k/dataset_manifest.json"
        if is_real_profile
        else root / "data/demo/dataset_manifest.json"
    )
    default_benchmark_report = (
        root / "artifacts/benchmarks/kuairand-pure-sample.json"
        if is_real_profile
        else root / "data/demo/benchmark.json"
    )
    dataset_manifest_path = Path(
        os.getenv(
            "STREAMRANK_DATASET_MANIFEST",
            default_dataset_manifest,
        )
    )
    benchmark_report_path = Path(
        os.getenv(
            "STREAMRANK_BENCHMARK_REPORT",
            default_benchmark_report,
        )
    )
    focused_report_path = Path(
        os.getenv(
            "STREAMRANK_FOCUSED_REPORT",
            root / "artifacts/sequence-ranking-real/report.json",
        )
    )
    focused_dataset_path = Path(
        os.getenv(
            "STREAMRANK_FOCUSED_DATASET_MANIFEST",
            root / "data/processed/kuairand-pure-5k/dataset_manifest.json",
        )
    )
    if not dataset_manifest_path.is_absolute():
        dataset_manifest_path = root / dataset_manifest_path
    if not benchmark_report_path.is_absolute():
        benchmark_report_path = root / benchmark_report_path
    if not focused_report_path.is_absolute():
        focused_report_path = root / focused_report_path
    if not focused_dataset_path.is_absolute():
        focused_dataset_path = root / focused_dataset_path
    if is_real_profile:
        dataset_payload = _load_json_file(dataset_manifest_path)
        benchmark_payload = _load_json_file(benchmark_report_path)
        expected = (
            int(dataset_payload.get("items", -1)),
            int(dataset_payload.get("interactions", -1)),
        )
        observed = (len(engine.catalog), engine.fitted_event_count)
        if expected != observed:
            raise ValueError("runtime profile mismatch between dataset manifest and bound catalog")
        if benchmark_payload.get("deployment_id") != manifest.deployment_id:
            raise ValueError("benchmark report deployment_id does not match active deployment")

    frontend_dir = root / "frontend"
    if frontend_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def dashboard() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

    @app.get("/api/focused")
    def focused_algorithm_report() -> dict[str, object]:
        """Curated algorithm-lab payload backed only by generated, auditable artifacts."""
        if not focused_report_path.is_file() or not focused_dataset_path.is_file():
            raise HTTPException(status_code=404, detail="focused experiment is unavailable")
        report = _read_json_object(focused_report_path)
        dataset = _read_json_object(focused_dataset_path)
        winner = str(report.get("winner", "unknown"))
        return {
            "dataset": dataset,
            "experiment": report,
            "serving": {
                "deployment_id": manifest.deployment_id,
                "bound_model_kind": model_config.get("kind", "unknown"),
                "bound_model_architecture": model_config.get("architecture", "unknown"),
                "offline_winner": winner,
                "offline_winner_bound": (
                    model_config.get("kind") == "focused_ranker"
                    and model_config.get("architecture") == winner
                ),
                "message": (
                    "离线胜出模型已接入在线服务"
                    if (
                        model_config.get("kind") == "focused_ranker"
                        and model_config.get("architecture") == winner
                    )
                    else "当前在线推荐仍是工程链路模型，尚未接入离线胜出模型"
                ),
            },
        }

    @app.get("/api/serving-users")
    def serving_users() -> dict[str, object]:
        """Return a small curated set of real profiles that the active serving catalog knows."""
        query_time_ms = engine.fit_cutoff_ms + 1
        ranked_users = sorted(
            engine.user_category_events,
            key=lambda user_id: (-len(state.history(user_id, before_ms=query_time_ms)), user_id),
        )[:20]
        profiles = []
        for user_id in ranked_users:
            history = state.history(user_id, before_ms=query_time_ms)
            affinity = engine._affinity(user_id, query_time_ms)
            profiles.append(
                {
                    "user_id": user_id,
                    "history_size": len(history),
                    "preferred_category": affinity.most_common(1)[0][0] if affinity else None,
                }
            )
        return {
            "profiles": profiles,
            "default_user_id": profiles[0]["user_id"] if profiles else None,
            "catalog_scope": index_config.get("fit_scope", "bound catalog log"),
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "deployment_id": engine.manifest.deployment_id,
            "catalog_items": len(engine.catalog),
            "fit_cutoff_ms": engine.fit_cutoff_ms,
            "state_backend": state_backend,
            "event_mode": event_mode,
            "serving_fit_scope": index_config.get("fit_scope", "bound catalog log"),
            "degraded_reason": engine.last_degraded_reason or None,
        }

    @app.get("/ready")
    def ready() -> dict[str, object]:
        if not engine.catalog:
            raise HTTPException(status_code=503, detail="catalog is empty")
        return {"ready": True, "deployment_id": engine.manifest.deployment_id}

    @app.get("/manifest")
    def get_manifest() -> DeploymentManifest:
        return engine.manifest

    @app.get("/recommend/{user_id}")
    def recommend(
        user_id: int, top_k: int = 20, query_time_ms: int | None = None, tab: int = 0
    ) -> dict[str, object]:
        effective_query_time_ms = query_time_ms or int(time.time() * 1000)
        trace: dict[str, object] = {}
        try:
            candidates = engine.recommend(
                user_id, min(max(1, top_k), 100), effective_query_time_ms, tab, trace=trace
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        history_size = len(state.history(user_id, before_ms=effective_query_time_ms))
        return {
            "user_id": user_id,
            "deployment_id": engine.manifest.deployment_id,
            "ranker": {
                "kind": model_config.get("kind", "unknown"),
                "architecture": model_config.get("architecture"),
            },
            "pipeline": {
                "catalog_size": trace.get("catalog_size", 0),
                "excluded_seen": trace.get("excluded_seen", 0),
                "stages": [
                    {"stage": "recall_itemcf", "count": trace.get("recall_itemcf", 0)},
                    {"stage": "recall_popularity", "count": trace.get("recall_popularity", 0)},
                    {"stage": "fusion", "count": trace.get("fused", 0)},
                    {"stage": "ranking", "count": trace.get("ranked", 0)},
                    {"stage": "rerank", "count": trace.get("returned", 0)},
                ],
            },
            "history_size": history_size,
            "personalization_mode": (
                "history-aware" if history_size else "cold-start-popularity-fallback"
            ),
            "items": [
                {
                    "item_id": candidate.item_id,
                    "score": candidate.rank_score,
                    "sources": sorted(candidate.retrieval_sources),
                    "source_ranks": candidate.source_ranks,
                    "features": {
                        "category": candidate.features.get("category"),
                        "is_click": candidate.features.get("is_click"),
                        "long_view": candidate.features.get("long_view"),
                        "is_like": candidate.features.get("is_like"),
                        "is_hate": candidate.features.get("is_hate"),
                    },
                }
                for candidate in candidates
            ],
        }

    @app.post("/events")
    def ingest(
        payload: EventPayload, x_event_id: str | None = Header(default=None)
    ) -> dict[str, object]:
        values = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        event = Interaction(**values)
        if producer is not None:
            producer.send(values, event_id=x_event_id)
            applied = engine.ingest(event, event_id=x_event_id)
            return {
                "accepted": True,
                "applied": applied,
                "deduplicated": not applied,
                "mode": "kafka+sync",
            }
        applied = engine.ingest(event, event_id=x_event_id)
        return {"applied": applied, "deduplicated": not applied, "mode": "sync"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return "\n".join(f"{name} {value}" for name, value in engine.metrics().items()) + "\n"

    return app


app = create_app()
