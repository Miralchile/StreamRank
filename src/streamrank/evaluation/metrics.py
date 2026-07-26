from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Iterable


def _validate(labels: Iterable[int], predictions: Iterable[float]) -> tuple[list[int], list[float]]:
    ys = [int(value) for value in labels]
    ps = [min(1 - 1e-15, max(1e-15, float(value))) for value in predictions]
    if len(ys) != len(ps) or not ys:
        raise ValueError("labels and predictions must be non-empty and equal length")
    if any(value not in (0, 1) for value in ys):
        raise ValueError("labels must be binary")
    return ys, ps


def roc_auc(labels: Iterable[int], predictions: Iterable[float]) -> float:
    ys, ps = _validate(labels, predictions)
    positives = sum(ys)
    negatives = len(ys) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both classes")
    ordered = sorted(zip(ps, ys, strict=True), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def pr_auc(labels: Iterable[int], predictions: Iterable[float]) -> float:
    ys, ps = _validate(labels, predictions)
    positives = sum(ys)
    if positives == 0:
        raise ValueError("PR-AUC requires positive labels")
    ordered = sorted(zip(ps, ys, strict=True), key=lambda pair: -pair[0])
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_labels = [label for _, label in ordered[index:end]]
        true_positives += sum(group_labels)
        false_positives += len(group_labels) - sum(group_labels)
        recall = true_positives / positives
        precision = true_positives / max(1, true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def log_loss(labels: Iterable[int], predictions: Iterable[float]) -> float:
    ys, ps = _validate(labels, predictions)
    return -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(ys, ps, strict=True)
    ) / len(ys)


def brier_score(labels: Iterable[int], predictions: Iterable[float]) -> float:
    ys, ps = _validate(labels, predictions)
    return sum((p - y) ** 2 for y, p in zip(ys, ps, strict=True)) / len(ys)


def expected_calibration_error(
    labels: Iterable[int], predictions: Iterable[float], bins: int = 10
) -> float:
    ys, ps = _validate(labels, predictions)
    bucketed: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for y, p in zip(ys, ps, strict=True):
        bucketed[min(bins - 1, int(p * bins))].append((y, p))
    error = 0.0
    for bucket in bucketed:
        if not bucket:
            continue
        accuracy = sum(y for y, _ in bucket) / len(bucket)
        confidence = sum(p for _, p in bucket) / len(bucket)
        error += len(bucket) / len(ys) * abs(accuracy - confidence)
    return error


def binary_classification_report(
    labels: Iterable[int], predictions: Iterable[float]
) -> dict[str, float]:
    ys, ps = _validate(labels, predictions)
    report = {
        "positive_rate": sum(ys) / len(ys),
        "log_loss": log_loss(ys, ps),
        "brier": brier_score(ys, ps),
        "ece_10": expected_calibration_error(ys, ps, bins=10),
    }
    if 0 < sum(ys) < len(ys):
        report["roc_auc"] = roc_auc(ys, ps)
        report["pr_auc"] = pr_auc(ys, ps)
    return report


def gauc(
    labels: Iterable[int],
    predictions: Iterable[float],
    group_ids: Iterable[Hashable],
    weighting: str = "impressions",
) -> dict[str, float]:
    ys, ps = _validate(labels, predictions)
    groups = list(group_ids)
    if len(groups) != len(ys):
        raise ValueError("group_ids length must equal labels length")
    grouped: defaultdict[Hashable, list[tuple[int, float]]] = defaultdict(list)
    for group, label, prediction in zip(groups, ys, ps, strict=True):
        grouped[group].append((label, prediction))
    numerator = 0.0
    denominator = 0.0
    valid_groups = 0
    valid_rows = 0
    for pairs in grouped.values():
        group_labels = [label for label, _ in pairs]
        if len(set(group_labels)) < 2:
            continue
        group_predictions = [prediction for _, prediction in pairs]
        weight = len(pairs) if weighting == "impressions" else 1.0
        numerator += weight * roc_auc(group_labels, group_predictions)
        denominator += weight
        valid_groups += 1
        valid_rows += len(pairs)
    return {
        "gauc": numerator / denominator if denominator else float("nan"),
        "valid_groups": float(valid_groups),
        "valid_group_fraction": valid_groups / max(1, len(grouped)),
        "valid_row_fraction": valid_rows / max(1, len(ys)),
        "weighting_impressions": float(weighting == "impressions"),
    }
