"""Time-correct offline evaluation of the retrieval stage.

Protocol: retrievers are fitted only on events before ``fit_end_ms``. For each
user the query state is their pre-cutoff trajectory; targets are items the user
positively engages with (``long_view``) inside the evaluation window and has
never interacted with before the cutoff. The candidate universe is the catalog
observed before the cutoff and previously seen items are excluded, mirroring
the serving engine. Reported numbers are next-positive retrieval quality inside
the logged candidate pool; they are not exposure-debiased.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from streamrank.domain import Interaction
from streamrank.retrieval.fusion import reciprocal_rank_fusion
from streamrank.retrieval.itemcf import ItemCFRetriever
from streamrank.retrieval.popularity import PopularityRetriever

HISTORY_LABELS = ("is_click", "long_view", "is_like")


@dataclass
class RecallQuery:
    user_id: int
    history: list[int]
    seen: set[int]
    preferred_category: str | None
    targets: set[int]


def build_queries(
    events: Sequence[Interaction], fit_end_ms: int, eval_end_ms: int
) -> list[RecallQuery]:
    history: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    affinity: dict[int, Counter] = defaultdict(Counter)
    targets: dict[int, set[int]] = defaultdict(set)
    for event in sorted(events, key=lambda row: row.event_time_ms):
        if event.event_time_ms < fit_end_ms:
            seen[event.user_id].add(event.item_id)
            if any(getattr(event, label) for label in HISTORY_LABELS):
                history[event.user_id].append(event.item_id)
            if event.long_view or event.is_like:
                affinity[event.user_id][event.category] += 1
        elif event.event_time_ms < eval_end_ms and event.long_view:
            targets[event.user_id].add(event.item_id)
    queries = []
    for user_id, target_items in targets.items():
        fresh_targets = target_items - seen[user_id]
        if not fresh_targets or not history[user_id]:
            continue
        preferred = affinity[user_id].most_common(1)[0][0] if affinity[user_id] else None
        queries.append(
            RecallQuery(
                user_id=user_id,
                history=history[user_id],
                seen=set(seen[user_id]),
                preferred_category=preferred,
                targets=fresh_targets,
            )
        )
    return queries


def evaluate_recall(
    events: Iterable[Interaction],
    fit_end_ms: int,
    eval_end_ms: int,
    ks: Sequence[int] = (50, 100, 200, 500),
) -> dict[str, object]:
    rows = list(events)
    fit_events = [event for event in rows if event.event_time_ms < fit_end_ms]
    if not fit_events:
        raise ValueError("no events before fit_end_ms")
    catalog = {event.item_id for event in fit_events}
    itemcf = ItemCFRetriever().fit(fit_events)
    popularity = PopularityRetriever().fit(fit_events)
    queries = build_queries(rows, fit_end_ms, eval_end_ms)
    if not queries:
        raise ValueError("no evaluable users with fresh targets in the window")
    max_k = max(ks)
    sources = ("itemcf", "popularity", "rrf_fusion")
    sums = {source: {k: {"recall": 0.0, "hit": 0.0} for k in ks} for source in sources}
    recommended: dict[str, set[int]] = {source: set() for source in sums}
    for query in queries:
        per_source = {
            "itemcf": [
                item_id
                for item_id, _ in itemcf.recommend(
                    query.history, top_k=max_k, exclude=query.seen, allowed_items=catalog
                )
            ],
            "popularity": [
                item_id
                for item_id, _ in (
                    popularity.recommend(
                        top_k=max_k,
                        category=query.preferred_category,
                        exclude=query.seen,
                        allowed_items=catalog,
                    )
                    or popularity.recommend(top_k=max_k, exclude=query.seen, allowed_items=catalog)
                )
            ],
        }
        fused = reciprocal_rank_fusion(
            {
                "itemcf": [(item_id, 1.0) for item_id in per_source["itemcf"]],
                "popularity": [(item_id, 1.0) for item_id in per_source["popularity"]],
            },
            top_k=max_k,
        )
        per_source["rrf_fusion"] = [candidate.item_id for candidate in fused]
        for source, items in per_source.items():
            recommended[source].update(items[:max_k])
            for k in ks:
                top = set(items[:k])
                overlap = len(top & query.targets)
                sums[source][k]["recall"] += overlap / len(query.targets)
                sums[source][k]["hit"] += 1.0 if overlap else 0.0
    users = len(queries)
    metrics = {
        source: {
            f"@{k}": {
                "recall": round(values[k]["recall"] / users, 4),
                "hit_rate": round(values[k]["hit"] / users, 4),
            }
            for k in ks
        }
        for source, values in sums.items()
    }
    for source in metrics:
        metrics[source]["catalog_coverage"] = round(len(recommended[source]) / len(catalog), 4)
    return {
        "protocol": {
            "fit_end_ms": fit_end_ms,
            "eval_end_ms": eval_end_ms,
            "target_label": "long_view",
            "history_labels": list(HISTORY_LABELS),
            "candidate_universe": (
                "items observed before fit_end_ms; previously seen items excluded"
            ),
            "caveat": (
                "next-positive retrieval inside the logged candidate pool; not exposure-debiased"
            ),
        },
        "fit_events": len(fit_events),
        "catalog_items": len(catalog),
        "evaluated_users": users,
        "ks": list(ks),
        "sources": metrics,
    }
