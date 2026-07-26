from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from streamrank.domain import Interaction


class ItemCFRetriever:
    """Time-safe ItemCF when fit only on events before the evaluation cutoff."""

    def __init__(self, positive_label: str = "long_view", max_user_items: int = 200):
        self.positive_label = positive_label
        self.max_user_items = max_user_items
        self.similarity: dict[int, dict[int, float]] = defaultdict(dict)

    def fit(self, events: Iterable[Interaction]) -> "ItemCFRetriever":
        histories: dict[int, list[int]] = defaultdict(list)
        item_frequency = Counter()
        for event in sorted(events, key=lambda row: row.event_time_ms):
            if getattr(event, self.positive_label):
                histories[event.user_id].append(event.item_id)
                item_frequency[event.item_id] += 1

        cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
        for items in histories.values():
            unique_items = list(dict.fromkeys(items[-self.max_user_items :]))
            if len(unique_items) < 2:
                continue
            contribution = 1.0 / math.log2(2 + len(unique_items))
            for left in unique_items:
                for right in unique_items:
                    if left != right:
                        cooccurrence[left][right] += contribution

        for left, neighbors in cooccurrence.items():
            for right, value in neighbors.items():
                denominator = math.sqrt(item_frequency[left] * item_frequency[right])
                self.similarity[left][right] = value / max(1.0, denominator)
        return self

    def recommend(
        self,
        history: list[int],
        top_k: int,
        exclude: set[int] | None = None,
        allowed_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        excluded = set(history) | (exclude or set())
        scores = Counter()
        for recency, item_id in enumerate(reversed(history[-50:]), start=1):
            recency_weight = 1.0 / math.log2(1 + recency)
            for neighbor, similarity in self.similarity.get(item_id, {}).items():
                if neighbor not in excluded:
                    scores[neighbor] += similarity * recency_weight
        return [
            (item_id, float(score))
            for item_id, score in scores.most_common()
            if allowed_items is None or item_id in allowed_items
        ][:top_k]
