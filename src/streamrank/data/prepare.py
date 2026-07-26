from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOG_FILES = (
    ("log_standard_4_08_to_4_21_pure.csv", "standard"),
    ("log_standard_4_22_to_5_08_pure.csv", "standard"),
    ("log_random_4_22_to_5_08_pure.csv", "random"),
)

OUTPUT_FIELDS = (
    "user_id",
    "item_id",
    "event_time_ms",
    "is_click",
    "long_view",
    "is_like",
    "is_hate",
    "is_follow",
    "tab",
    "is_rand",
    "logging_policy",
    "request_id",
    "category",
    "author_id",
    "upload_time_ms",
)


def _stable_user_score(user_id: int, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{user_id}".encode()).digest()


def select_users(user_features_path: Path, limit: int, seed: int) -> tuple[set[int], int]:
    with user_features_path.open("r", encoding="utf-8", newline="") as handle:
        user_ids = [int(row["user_id"]) for row in csv.DictReader(handle)]
    if limit <= 0 or limit >= len(user_ids):
        return set(user_ids), len(user_ids)
    selected = sorted(user_ids, key=lambda user_id: _stable_user_score(user_id, seed))[:limit]
    return set(selected), len(user_ids)


def _upload_time_ms(raw: str) -> int:
    if not raw:
        return 0
    # Dataset dates have day precision. Midnight Asia/Shanghai is a documented approximation,
    # not a claim about the exact upload instant.
    value = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=8)))
    return int(value.timestamp() * 1000)


def load_item_metadata(path: Path) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = int(row["video_id"])
            tags = [value.strip() for value in (row.get("tag") or "").split(",") if value.strip()]
            metadata[item_id] = {
                "category": f"tag:{tags[0]}" if tags else "UNKNOWN",
                "tags": tags,
                "author_id": int(float(row.get("author_id") or -1)),
                "upload_time_ms": _upload_time_ms(row.get("upload_dt") or ""),
                "upload_date_precision": "day_asia_shanghai_midnight_approximation",
                "video_type": row.get("video_type") or "UNKNOWN",
                "duration_ms": int(float(row.get("video_duration") or 0)),
            }
    return metadata


def _copy_selected_users(source: Path, destination: Path, users: set[int]) -> int:
    with (
        source.open("r", encoding="utf-8", newline="") as reader_handle,
        destination.open("w", encoding="utf-8", newline="") as writer_handle,
    ):
        reader = csv.DictReader(reader_handle)
        writer = csv.DictWriter(writer_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            if int(row["user_id"]) in users:
                writer.writerow(row)
                count += 1
    return count


def prepare_kuairand_pure(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    users: int = 500,
    seed: int = 2026,
    archive_md5: str | None = None,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    required = [raw_dir / name for name, _ in LOG_FILES] + [
        raw_dir / "user_features_pure.csv",
        raw_dir / "video_features_basic_pure.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing KuaiRand-Pure files: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_users, source_user_count = select_users(
        raw_dir / "user_features_pure.csv", users, seed
    )
    item_metadata = load_item_metadata(raw_dir / "video_features_basic_pure.csv")
    selected_user_rows = _copy_selected_users(
        raw_dir / "user_features_pure.csv",
        output_dir / "user_features_selected.csv",
        selected_users,
    )

    output_path = output_dir / "interactions.csv"
    raw_rows: Counter[str] = Counter()
    output_rows: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    tabs: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    items: set[int] = set()
    min_time_ms: int | None = None
    max_time_ms: int | None = None
    metadata_hits = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for filename, policy in LOG_FILES:
            with (raw_dir / filename).open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_rows[policy] += 1
                    user_id = int(row["user_id"])
                    if user_id not in selected_users:
                        continue
                    item_id = int(row["video_id"])
                    item = item_metadata.get(item_id, {})
                    event_time_ms = int(row["time_ms"])
                    output = {
                        "user_id": user_id,
                        "item_id": item_id,
                        "event_time_ms": event_time_ms,
                        "is_click": int(row["is_click"]),
                        "long_view": int(row["long_view"]),
                        "is_like": int(row["is_like"]),
                        "is_hate": int(row["is_hate"]),
                        "is_follow": int(row["is_follow"]),
                        "tab": int(row["tab"]),
                        "is_rand": int(row["is_rand"]),
                        "logging_policy": policy,
                        # KuaiRand-Pure does not expose a request/slate identifier.
                        "request_id": "",
                        "category": item.get("category", "UNKNOWN"),
                        "author_id": item.get("author_id", -1),
                        "upload_time_ms": item.get("upload_time_ms", 0),
                    }
                    writer.writerow(output)
                    output_rows[policy] += 1
                    items.add(item_id)
                    dates[row["date"]] += 1
                    tabs[row["tab"]] += 1
                    metadata_hits += int(bool(item))
                    for label in ("is_click", "long_view", "is_like", "is_hate"):
                        labels[label] += int(output[label])
                    min_time_ms = (
                        event_time_ms if min_time_ms is None else min(min_time_ms, event_time_ms)
                    )
                    max_time_ms = (
                        event_time_ms if max_time_ms is None else max(max_time_ms, event_time_ms)
                    )

    used_items = {
        item_id: item_metadata[item_id] for item_id in sorted(items) if item_id in item_metadata
    }
    (output_dir / "item_features.json").write_text(
        json.dumps(used_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    row_count = sum(output_rows.values())
    report = {
        "dataset": "KuaiRand-Pure",
        "mode": "full-user-cohort" if users <= 0 else "deterministic-user-sample",
        "license": "CC-BY-SA-4.0",
        "official_source": "https://zenodo.org/records/10439422",
        "archive_md5": archive_md5,
        "seed": seed,
        "selected_users": len(selected_users),
        "source_users": source_user_count,
        "selected_user_feature_rows": selected_user_rows,
        "items": len(items),
        "interactions": row_count,
        "raw_rows_scanned": dict(raw_rows),
        "rows_by_logging_policy": dict(output_rows),
        "label_positive_rates": {label: labels[label] / max(1, row_count) for label in labels},
        "tab_counts": dict(sorted(tabs.items(), key=lambda pair: int(pair[0]))),
        "date_counts": dict(sorted(dates.items())),
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "item_metadata_coverage": metadata_hits / max(1, row_count),
        "item_metadata_source": "video_features_basic_pure.csv",
        "upload_time_semantics": "upload_dt parsed as Asia/Shanghai midnight; day precision only",
        "request_group_semantics": "approximate user + time window; source has no request_id",
        "sequence_scope": "candidate-pool interactions only; not complete user history",
        "excluded_feature_source": "video_features_statistic_pure.csv (not point-in-time)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
