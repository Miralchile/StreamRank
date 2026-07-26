from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from streamrank.data.loader import load_interactions
from streamrank.domain import Interaction

TASKS = ("is_click", "long_view", "is_like", "is_hate")


def iso_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


@dataclass(frozen=True)
class Vocabulary:
    users: dict[int, int]
    items: dict[int, int]

    @property
    def num_users(self) -> int:
        # 0 is padding, 1 is OOV.
        return len(self.users) + 2

    @property
    def num_items(self) -> int:
        return len(self.items) + 2


@dataclass
class SplitArrays:
    user_ids: np.ndarray
    item_ids: np.ndarray
    tab_ids: np.ndarray
    histories: np.ndarray
    history_mask: np.ndarray
    numeric: np.ndarray
    labels: dict[str, np.ndarray]
    raw_user_ids: np.ndarray
    event_time_ms: np.ndarray
    logging_policy: np.ndarray

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])


@dataclass
class PreparedDataset:
    vocabulary: Vocabulary
    splits: dict[str, SplitArrays]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    metadata: dict[str, Any]


def _build_vocabulary(events: Iterable[Interaction], train_end_ms: int) -> Vocabulary:
    train = [
        event
        for event in events
        if event.logging_policy == "standard" and event.event_time_ms < train_end_ms
    ]
    users = {value: index + 2 for index, value in enumerate(sorted({e.user_id for e in train}))}
    items = {value: index + 2 for index, value in enumerate(sorted({e.item_id for e in train}))}
    return Vocabulary(users=users, items=items)


def _split_name(
    event: Interaction, train_end: int, validation_end: int, test_end: int
) -> str | None:
    if event.logging_policy == "random":
        return "random_diagnostic" if event.event_time_ms < test_end else None
    if event.event_time_ms < train_end:
        return "train"
    if event.event_time_ms < validation_end:
        return "validation"
    if event.event_time_ms < test_end:
        return "test"
    return None


def prepare_sequence_dataset(csv_path: str | Path, config: dict[str, Any]) -> PreparedDataset:
    """Build candidate-aware histories with predict-all-then-update time-bucket semantics."""
    events = sorted(
        load_interactions(csv_path),
        key=lambda event: (event.event_time_ms, event.user_id, event.item_id, event.is_rand),
    )
    split_config = config["splits"]
    train_end = iso_to_ms(split_config["train_end"])
    validation_end = iso_to_ms(split_config["validation_end"])
    test_end = iso_to_ms(split_config["test_end"])
    vocabulary = _build_vocabulary(events, train_end)
    max_history = int(config.get("max_history", 50))
    request_window_ms = int(config.get("request_group_window_ms", 1000))
    max_rows = {key: int(value) for key, value in config.get("max_rows_per_split", {}).items()}

    rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    histories: dict[int, list[int]] = defaultdict(list)
    user_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    item_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])

    def append(event: Interaction) -> None:
        split = _split_name(event, train_end, validation_end, test_end)
        if split is None or (
            max_rows.get(split, 0) and len(rows[split]["user_ids"]) >= max_rows[split]
        ):
            return
        history = histories[event.user_id][-max_history:]
        mapped_history = [vocabulary.items.get(item_id, 1) for item_id in history]
        padding = max_history - len(mapped_history)
        user_count, user_clicks, user_long = user_stats[event.user_id]
        item_count, item_clicks, item_long = item_stats[event.item_id]
        age_days = (
            max(0.0, (event.event_time_ms - event.upload_time_ms) / 86_400_000)
            if event.upload_time_ms
            else 0.0
        )
        numeric = [
            math.log1p(user_count),
            math.log1p(item_count),
            user_clicks / max(1, user_count),
            user_long / max(1, user_count),
            item_clicks / max(1, item_count),
            item_long / max(1, item_count),
            event.tab / 14.0 if event.tab >= 0 else -1.0,
            math.log1p(age_days),
        ]
        target = rows[split]
        target["user_ids"].append(vocabulary.users.get(event.user_id, 1))
        target["item_ids"].append(vocabulary.items.get(event.item_id, 1))
        target["tab_ids"].append(event.tab + 1 if 0 <= event.tab <= 14 else 0)
        target["histories"].append([0] * padding + mapped_history)
        target["history_mask"].append([False] * padding + [True] * len(mapped_history))
        target["numeric"].append(numeric)
        target["raw_user_ids"].append(event.user_id)
        target["event_time_ms"].append(event.event_time_ms)
        target["logging_policy"].append(event.logging_policy)
        for task in TASKS:
            target[task].append(getattr(event, task))

    # A time bucket approximates a request boundary. All rows in it are materialized before any
    # feedback is consumed, avoiding within-request label leakage.
    bucket: list[Interaction] = []
    current_bucket: int | None = None
    for event in events:
        event_bucket = event.event_time_ms // request_window_ms
        if current_bucket is not None and event_bucket != current_bucket:
            for pending in bucket:
                append(pending)
            for pending in bucket:
                stats = user_stats[pending.user_id]
                stats[0] += 1
                stats[1] += pending.is_click
                stats[2] += pending.long_view
                item = item_stats[pending.item_id]
                item[0] += 1
                item[1] += pending.is_click
                item[2] += pending.long_view
                if pending.is_click or pending.long_view or pending.is_like:
                    histories[pending.user_id].append(pending.item_id)
            bucket = []
        bucket.append(event)
        current_bucket = event_bucket
    if bucket:
        for pending in bucket:
            append(pending)

    numeric_train = np.asarray(rows["train"]["numeric"], dtype=np.float32)
    mean = numeric_train.mean(axis=0)
    std = numeric_train.std(axis=0)
    std[std < 1e-6] = 1.0

    splits: dict[str, SplitArrays] = {}
    for name, values in rows.items():
        numeric = (np.asarray(values["numeric"], dtype=np.float32) - mean) / std
        splits[name] = SplitArrays(
            user_ids=np.asarray(values["user_ids"], dtype=np.int64),
            item_ids=np.asarray(values["item_ids"], dtype=np.int64),
            tab_ids=np.asarray(values["tab_ids"], dtype=np.int64),
            histories=np.asarray(values["histories"], dtype=np.int64),
            history_mask=np.asarray(values["history_mask"], dtype=np.bool_),
            numeric=numeric,
            labels={task: np.asarray(values[task], dtype=np.float32) for task in TASKS},
            raw_user_ids=np.asarray(values["raw_user_ids"], dtype=np.int64),
            event_time_ms=np.asarray(values["event_time_ms"], dtype=np.int64),
            logging_policy=np.asarray(values["logging_policy"]),
        )
    return PreparedDataset(
        vocabulary=vocabulary,
        splits=splits,
        numeric_mean=mean,
        numeric_std=std,
        metadata={
            "source": str(csv_path),
            "sequence_scope": "KuaiRand-Pure candidate-pool history; not complete user history",
            "request_semantics": f"global {request_window_ms}ms time-bucket approximation",
            "rows": {name: len(split) for name, split in splits.items()},
            "train_users": len(vocabulary.users),
            "train_items": len(vocabulary.items),
            "max_history": max_history,
        },
    )


class TorchSequenceDataset:
    def __init__(self, arrays: SplitArrays):
        import torch

        self.tensors = {
            "user_ids": torch.from_numpy(arrays.user_ids),
            "item_ids": torch.from_numpy(arrays.item_ids),
            "tab_ids": torch.from_numpy(arrays.tab_ids),
            "histories": torch.from_numpy(arrays.histories),
            "history_mask": torch.from_numpy(arrays.history_mask),
            "numeric": torch.from_numpy(arrays.numeric),
            **{task: torch.from_numpy(values) for task, values in arrays.labels.items()},
        }

    def __len__(self) -> int:
        return int(self.tensors["user_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {key: value[index] for key, value in self.tensors.items()}
