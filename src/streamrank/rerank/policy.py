from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from streamrank.domain import Candidate


@dataclass(frozen=True)
class DiversityPolicy:
    top_k: int = 20
    max_per_category: int = 5
    max_per_author: int = 3
    concentration_penalty: float = 0.05


def rerank(candidates: Iterable[Candidate], policy: DiversityPolicy) -> list[Candidate]:
    remaining = list(candidates)
    selected: list[Candidate] = []
    category_counts = Counter()
    author_counts = Counter()

    while remaining and len(selected) < policy.top_k:
        best: Candidate | None = None
        best_adjusted = float("-inf")
        for candidate in remaining:
            category = str(candidate.features.get("category", "UNKNOWN"))
            author = int(candidate.features.get("author_id", -1))
            if category_counts[category] >= policy.max_per_category:
                continue
            if author >= 0 and author_counts[author] >= policy.max_per_author:
                continue
            penalty = policy.concentration_penalty * (
                category_counts[category] + (author_counts[author] if author >= 0 else 0)
            )
            adjusted = candidate.rank_score - penalty
            if adjusted > best_adjusted:
                best, best_adjusted = candidate, adjusted
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        category_counts[str(best.features.get("category", "UNKNOWN"))] += 1
        author = int(best.features.get("author_id", -1))
        if author >= 0:
            author_counts[author] += 1
    return selected
