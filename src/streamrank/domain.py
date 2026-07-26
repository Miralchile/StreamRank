from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    event_time_ms: int
    is_click: int = 0
    long_view: int = 0
    is_like: int = 0
    is_hate: int = 0
    is_follow: int = 0
    tab: int = -1
    is_rand: int = 0
    logging_policy: str = "standard"
    request_id: str | None = None
    category: str = "UNKNOWN"
    author_id: int = -1
    upload_time_ms: int = 0

    @property
    def labels(self) -> dict[str, int]:
        return {
            "is_click": self.is_click,
            "long_view": self.long_view,
            "is_like": self.is_like,
            "is_hate": self.is_hate,
        }


@dataclass
class Candidate:
    item_id: int
    retrieval_sources: set[str] = field(default_factory=set)
    source_scores: dict[str, float] = field(default_factory=dict)
    source_ranks: dict[str, int] = field(default_factory=dict)
    features: dict[str, float | str | int] = field(default_factory=dict)
    rank_score: float = 0.0

    def add_source(self, source: str, score: float, rank: int) -> None:
        self.retrieval_sources.add(source)
        self.source_scores[source] = float(score)
        self.source_ranks[source] = int(rank)


@dataclass(frozen=True)
class Prediction:
    interaction: Interaction
    probabilities: Mapping[str, float]
    score: float
    deployment_id: str
