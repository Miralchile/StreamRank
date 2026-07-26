from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from streamrank.domain import Candidate


def reciprocal_rank_fusion(
    results: Mapping[str, Sequence[tuple[int, float]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[Candidate]:
    """Fuse ranks without pretending raw scores from different retrievers are comparable."""
    candidates: dict[int, Candidate] = {}
    fused_scores: defaultdict[int, float] = defaultdict(float)
    for source, ranked_items in results.items():
        for rank, (item_id, raw_score) in enumerate(ranked_items, start=1):
            candidate = candidates.setdefault(item_id, Candidate(item_id=item_id))
            candidate.add_source(source, raw_score, rank)
            fused_scores[item_id] += 1.0 / (rrf_k + rank)

    ordered = sorted(candidates.values(), key=lambda row: (-fused_scores[row.item_id], row.item_id))
    for candidate in ordered:
        candidate.features["rrf_score"] = fused_scores[candidate.item_id]
        candidate.features["retrieval_source_count"] = len(candidate.retrieval_sources)
        for source, rank in candidate.source_ranks.items():
            candidate.features[f"rank_{source}"] = rank
            candidate.features[f"score_{source}"] = candidate.source_scores[source]
    return ordered[:top_k]
