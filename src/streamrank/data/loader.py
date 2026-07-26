from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator

from streamrank.domain import Interaction

ALIASES = {
    "video_id": "item_id",
    "time_ms": "event_time_ms",
}


def _int(row: dict[str, str], name: str, default: int = 0) -> int:
    raw = row.get(name, "")
    if raw in (None, ""):
        return default
    return int(float(raw))


def load_interactions(path: str | Path, logging_policy: str | None = None) -> Iterator[Interaction]:
    """Load KuaiRand-like CSV without assuming that tab identifies the UI type."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {ALIASES.get(k, k): v for k, v in row.items()}
            inferred_policy = (
                logging_policy
                or normalized.get("logging_policy")
                or ("random" if _int(normalized, "is_rand") else "standard")
            )
            yield Interaction(
                user_id=_int(normalized, "user_id"),
                item_id=_int(normalized, "item_id"),
                event_time_ms=_int(normalized, "event_time_ms"),
                is_click=_int(normalized, "is_click"),
                long_view=_int(normalized, "long_view"),
                is_like=_int(normalized, "is_like"),
                is_hate=_int(normalized, "is_hate"),
                is_follow=_int(normalized, "is_follow"),
                tab=_int(normalized, "tab", -1),
                is_rand=_int(normalized, "is_rand"),
                logging_policy=inferred_policy,
                request_id=normalized.get("request_id") or None,
                category=normalized.get("category") or "UNKNOWN",
                author_id=_int(normalized, "author_id", -1),
                upload_time_ms=_int(normalized, "upload_time_ms", 0),
            )


def merge_by_event_time(*streams: Iterable[Interaction]) -> list[Interaction]:
    events = [event for stream in streams for event in stream]
    return sorted(events, key=lambda event: (event.event_time_ms, event.user_id, event.item_id))
