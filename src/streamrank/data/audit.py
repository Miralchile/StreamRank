from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from streamrank.domain import Interaction


@dataclass
class AuditReport:
    rows: int
    users: int
    items: int
    min_time_ms: int | None
    max_time_ms: int | None
    label_positive_rates: dict[str, float]
    tab_counts: dict[int, int]
    logging_policy_counts: dict[str, int]
    duplicate_event_keys: int
    invalid_binary_labels: int
    long_view_without_click: int
    like_without_click: int
    request_id_coverage: float
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_interactions(events: Iterable[Interaction]) -> AuditReport:
    materialized = list(events)
    labels = ("is_click", "long_view", "is_like", "is_hate")
    label_sums = Counter()
    tabs = Counter()
    policies = Counter()
    keys = Counter()
    invalid = 0
    long_without_click = 0
    like_without_click = 0
    request_ids = 0

    for event in materialized:
        tabs[event.tab] += 1
        policies[event.logging_policy] += 1
        keys[(event.user_id, event.item_id, event.event_time_ms, event.logging_policy)] += 1
        request_ids += int(event.request_id is not None)
        for label in labels:
            value = getattr(event, label)
            label_sums[label] += value
            invalid += int(value not in (0, 1))
        long_without_click += int(event.long_view == 1 and event.is_click == 0)
        like_without_click += int(event.is_like == 1 and event.is_click == 0)

    rows = len(materialized)
    warnings = [
        (
            "is_click has UI-dependent click/valid-play semantics; "
            "tab alone does not prove the UI layout."
        ),
        "video_features_statistic.csv is not a point-in-time snapshot and is excluded by default.",
        "visible_status is a current snapshot and must not be treated as historical availability.",
    ]
    if rows and long_without_click:
        warnings.append("long_view does not globally imply is_click in the observed data.")
    if rows and not request_ids:
        warnings.append(
            "request_id is absent; request grouping must be treated as an approximation."
        )

    return AuditReport(
        rows=rows,
        users=len({event.user_id for event in materialized}),
        items=len({event.item_id for event in materialized}),
        min_time_ms=min((event.event_time_ms for event in materialized), default=None),
        max_time_ms=max((event.event_time_ms for event in materialized), default=None),
        label_positive_rates={
            label: (label_sums[label] / rows if rows else 0.0) for label in labels
        },
        tab_counts=dict(sorted(tabs.items())),
        logging_policy_counts=dict(sorted(policies.items())),
        duplicate_event_keys=sum(count - 1 for count in keys.values() if count > 1),
        invalid_binary_labels=invalid,
        long_view_without_click=long_without_click,
        like_without_click=like_without_click,
        request_id_coverage=(request_ids / rows if rows else 0.0),
        warnings=warnings,
    )


def label_cross_table(events: Iterable[Interaction], left: str, right: str) -> dict[str, int]:
    table: defaultdict[str, int] = defaultdict(int)
    for event in events:
        table[f"{left}={getattr(event, left)},{right}={getattr(event, right)}"] += 1
    return dict(sorted(table.items()))
