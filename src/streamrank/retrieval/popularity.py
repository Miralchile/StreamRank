from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from streamrank.domain import Interaction


class PopularityRetriever:
    def __init__(self, positive_label: str = "long_view"):
        self.positive_label = positive_label
        self.global_counts: Counter[int] = Counter()
        self.category_counts: dict[str, Counter[int]] = defaultdict(Counter)

    def fit(self, events: Iterable[Interaction]) -> "PopularityRetriever":
        for event in events:
            if getattr(event, self.positive_label):
                self.global_counts[event.item_id] += 1
                self.category_counts[event.category][event.item_id] += 1
        return self

    def recommend(
        self,
        top_k: int,
        category: str | None = None,
        exclude: set[int] | None = None,
        allowed_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        excluded = exclude or set()
        source = self.category_counts.get(category, Counter()) if category else self.global_counts
        ranked = []
        for item_id, count in source.most_common():
            if item_id in excluded:
                continue
            if allowed_items is not None and item_id not in allowed_items:
                continue
            ranked.append((item_id, float(count)))
            if len(ranked) == top_k:
                break
        return ranked
