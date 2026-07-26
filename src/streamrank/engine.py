from __future__ import annotations

import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from streamrank.domain import Candidate, Interaction
from streamrank.online.state import InMemoryOnlineState, RedisOnlineState
from streamrank.ranking.scoring import PolicyScorer
from streamrank.ranking.serving_ranker import FocusedServingRanker
from streamrank.rerank.policy import DiversityPolicy, rerank
from streamrank.retrieval.fusion import reciprocal_rank_fusion
from streamrank.retrieval.itemcf import ItemCFRetriever
from streamrank.retrieval.popularity import PopularityRetriever
from streamrank.serving.manifest import DeploymentManifest


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass(frozen=True)
class CatalogItem:
    item_id: int
    category: str
    author_id: int
    upload_time_ms: int
    first_seen_time_ms: int


class RecommendationEngine:
    """Local engine whose ranking predictor can be replaced through a stable contract."""

    def __init__(
        self,
        manifest: DeploymentManifest,
        scorer: PolicyScorer,
        state: InMemoryOnlineState | RedisOnlineState | None = None,
        model_config: dict[str, object] | None = None,
        rerank_config: dict[str, object] | None = None,
    ):
        self.manifest = manifest
        self.scorer = scorer
        self.state = state or InMemoryOnlineState()
        self.model_config = model_config or {}
        self.rerank_config = rerank_config or {}
        self.popularity = PopularityRetriever()
        self.itemcf = ItemCFRetriever()
        self.catalog: dict[int, CatalogItem] = {}
        self.user_category_events: dict[int, list[tuple[int, str]]] = defaultdict(list)
        self.user_stats: dict[int, tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
        self.item_stats: dict[int, tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
        self.focused_ranker = (
            FocusedServingRanker.from_config(self.model_config)
            if self.model_config.get("kind") == "focused_ranker"
            else None
        )
        self.fit_cutoff_ms = 0
        self.fitted_event_count = 0
        self.request_count = 0
        self.failure_count = 0
        self.degraded_count = 0
        self.last_latency_ms = 0.0
        self.latencies_ms: deque[float] = deque(maxlen=10_000)
        self.last_degraded_reason = ""

    def fit(self, events: Iterable[Interaction]) -> "RecommendationEngine":
        rows = list(events)
        self.fitted_event_count = len(rows)
        self.fit_cutoff_ms = max((row.event_time_ms for row in rows), default=0)
        self.popularity.fit(rows)
        self.itemcf.fit(rows)
        for event in sorted(rows, key=lambda row: row.event_time_ms):
            existing = self.catalog.get(event.item_id)
            first_seen = (
                min(existing.first_seen_time_ms, event.event_time_ms)
                if existing
                else event.event_time_ms
            )
            self.catalog[event.item_id] = CatalogItem(
                event.item_id, event.category, event.author_id, event.upload_time_ms, first_seen
            )
            self.state.apply(event)
            user_count, user_clicks, user_long = self.user_stats[event.user_id]
            self.user_stats[event.user_id] = (
                user_count + 1,
                user_clicks + event.is_click,
                user_long + event.long_view,
            )
            item_count, item_clicks, item_long = self.item_stats[event.item_id]
            self.item_stats[event.item_id] = (
                item_count + 1,
                item_clicks + event.is_click,
                item_long + event.long_view,
            )
            if event.long_view or event.is_like:
                self.user_category_events[event.user_id].append(
                    (event.event_time_ms, event.category)
                )
        return self

    def _affinity(self, user_id: int, query_time_ms: int) -> Counter[str]:
        return Counter(
            category
            for event_time_ms, category in self.user_category_events[user_id]
            if event_time_ms < query_time_ms
        )

    def _probabilities(self, candidate: Candidate, affinity: Counter[str]) -> dict[str, float]:
        source_count = float(candidate.features.get("retrieval_source_count", 1))
        popularity = float(candidate.source_scores.get("popularity", 0.0))
        category = str(candidate.features.get("category", "UNKNOWN"))
        category_affinity = affinity[category]
        config = self.model_config
        base = (
            float(config.get("base_bias", -2.2))
            + float(config.get("popularity_weight", 0.35)) * math.log1p(popularity)
            + float(config.get("source_count_weight", 0.25)) * source_count
            + float(config.get("affinity_weight", 0.15)) * math.log1p(category_affinity)
        )
        return {
            "is_click": _sigmoid(base),
            "long_view": _sigmoid(base + float(config.get("long_view_offset", -0.45))),
            "is_like": _sigmoid(base + float(config.get("like_offset", -1.8))),
            "is_hate": _sigmoid(
                float(config.get("hate_bias", -4.0))
                + float(config.get("hate_affinity_weight", -0.1)) * category_affinity
            ),
        }

    def recommend(
        self,
        user_id: int,
        top_k: int = 20,
        query_time_ms: int | None = None,
        tab: int = 0,
        trace: dict[str, object] | None = None,
    ) -> list[Candidate]:
        """Run the multi-stage serving path; optionally record per-stage candidate counts."""
        started = time.perf_counter()
        self.request_count += 1
        try:
            query_time_ms = query_time_ms or int(time.time() * 1000)
            if self.fit_cutoff_ms and query_time_ms < self.fit_cutoff_ms:
                raise ValueError(
                    "query_time_ms precedes the fitted artifact cutoff; use the PIT replay engine"
                )
            history = self.state.history(user_id, before_ms=query_time_ms)
            allowed_items = {
                item_id
                for item_id, item in self.catalog.items()
                if max(item.upload_time_ms, item.first_seen_time_ms) <= query_time_ms
            }
            excluded = set(history)
            preferred_category = None
            affinity = self._affinity(user_id, query_time_ms)
            if affinity:
                preferred_category = affinity.most_common(1)[0][0]
            results = {
                "itemcf": self.itemcf.recommend(
                    history,
                    top_k=max(100, top_k * 5),
                    exclude=excluded,
                    allowed_items=allowed_items,
                ),
                "popularity": self.popularity.recommend(
                    top_k=max(100, top_k * 5),
                    category=preferred_category,
                    exclude=excluded,
                    allowed_items=allowed_items,
                ),
            }
            if not results["popularity"]:
                results["popularity"] = self.popularity.recommend(
                    top_k=max(100, top_k * 5),
                    exclude=excluded,
                    allowed_items=allowed_items,
                )
            candidates = reciprocal_rank_fusion(results, top_k=max(200, top_k * 10))
            if trace is not None:
                trace.update(
                    {
                        "catalog_size": len(allowed_items),
                        "excluded_seen": len(excluded),
                        "recall_itemcf": len(results["itemcf"]),
                        "recall_popularity": len(results["popularity"]),
                        "fused": len(candidates),
                        "ranked": len(candidates),
                        "returned": 0,
                    }
                )
            if not candidates:
                self.degraded_count += 1
                self.last_degraded_reason = "no_unseen_candidates"
                return []
            for candidate in candidates:
                item = self.catalog[candidate.item_id]
                candidate.features.update(
                    {
                        "category": item.category,
                        "author_id": item.author_id,
                        "upload_time_ms": item.upload_time_ms,
                    }
                )
                age_days = (
                    max(0.0, (query_time_ms - item.upload_time_ms) / 86_400_000)
                    if item.upload_time_ms
                    else 365.0
                )
                freshness = 1.0 / (1.0 + age_days)
                if self.focused_ranker is None:
                    candidate.features.update(self._probabilities(candidate, affinity))
                candidate.features["freshness"] = freshness
            if self.focused_ranker is not None:
                predictions = self.focused_ranker.predict(
                    user_id=user_id,
                    history=history,
                    candidates=candidates,
                    query_time_ms=query_time_ms,
                    tab=tab,
                    user_stats=self.user_stats,
                    item_stats=self.item_stats,
                )
                for candidate, probabilities in zip(candidates, predictions, strict=True):
                    candidate.features.update(probabilities)
            for candidate in candidates:
                freshness = float(candidate.features.get("freshness", 0.0))
                candidate.rank_score = self.scorer.score(candidate.features, freshness)
            reranked = rerank(
                candidates,
                DiversityPolicy(
                    top_k=top_k,
                    max_per_category=int(self.rerank_config.get("max_per_category", 5)),
                    max_per_author=int(self.rerank_config.get("max_per_author", 3)),
                    concentration_penalty=float(
                        self.rerank_config.get("concentration_penalty", 0.05)
                    ),
                ),
            )
            if trace is not None:
                trace["returned"] = len(reranked)
            return reranked
        except Exception:
            self.failure_count += 1
            raise
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.latencies_ms.append(self.last_latency_ms)

    def ingest(self, event: Interaction, event_id: str | None = None) -> bool:
        applied = self.state.apply(event, event_id=event_id, now_ms=int(time.time() * 1000))
        if applied and (event.long_view or event.is_like):
            self.user_category_events[event.user_id].append((event.event_time_ms, event.category))
        if applied:
            user_count, user_clicks, user_long = self.user_stats[event.user_id]
            self.user_stats[event.user_id] = (
                user_count + 1,
                user_clicks + event.is_click,
                user_long + event.long_view,
            )
            item_count, item_clicks, item_long = self.item_stats[event.item_id]
            self.item_stats[event.item_id] = (
                item_count + 1,
                item_clicks + event.is_click,
                item_long + event.long_view,
            )
        return applied

    def metrics(self) -> dict[str, float]:
        ordered = sorted(self.latencies_ms)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

        return {
            "streamrank_requests_total": float(self.request_count),
            "streamrank_failures_total": float(self.failure_count),
            "streamrank_last_latency_ms": self.last_latency_ms,
            "streamrank_latency_p50_ms": percentile(0.50),
            "streamrank_latency_p95_ms": percentile(0.95),
            "streamrank_latency_p99_ms": percentile(0.99),
            "streamrank_consumer_lag_ms": float(self.state.consumer_lag_ms),
            "streamrank_degraded_total": float(self.degraded_count),
        }
