"""Statistical support for the sequence-ranking model comparison.

Two complementary analyses:

1. ``bootstrap``: paired, user-clustered bootstrap on one seed's exported
   checkpoints. Users are resampled with replacement; per resample the
   selection score (mean of is_click and long_view impression-weighted user
   GAUC) is recomputed for every model from fixed per-user statistics, so the
   reported interval reflects sampling uncertainty of the evaluation cohort
   for this training run. It does not measure training-seed variance.
2. ``collect-seeds``: aggregates validation selection GAUC across independent
   seed reruns to check whether the model ranking is stable under retraining.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from streamrank.evaluation.metrics import roc_auc
from streamrank.focused.dataset import prepare_sequence_dataset
from streamrank.focused.models import FocusedRanker
from streamrank.focused.runner import _predict

PRIMARY_TASKS = ("is_click", "long_view")


def _per_user_auc(labels: np.ndarray, scores: np.ndarray, users: np.ndarray):
    """Return aligned arrays (auc, weight) over the split's unique users.

    Users without both label classes get weight 0 and are excluded from the
    weighted mean, matching the GAUC definition used for model selection.
    """
    order = np.argsort(users, kind="stable")
    labels, scores, users = labels[order], scores[order], users[order]
    unique_users, starts = np.unique(users, return_index=True)
    bounds = list(starts) + [len(users)]
    aucs = np.zeros(len(unique_users))
    weights = np.zeros(len(unique_users))
    for index in range(len(unique_users)):
        left, right = bounds[index], bounds[index + 1]
        chunk_labels = labels[left:right]
        if 0 < chunk_labels.sum() < len(chunk_labels):
            aucs[index] = roc_auc(chunk_labels.astype(int).tolist(), scores[left:right].tolist())
            weights[index] = right - left
    return unique_users, aucs, weights


def _selection_matrix(stats, resample_index):
    """Weighted GAUC mean for one task under a (B, U) user resample."""
    aucs, weights = stats
    drawn_auc = aucs[resample_index]
    drawn_weight = weights[resample_index]
    total = drawn_weight.sum(axis=1)
    total[total == 0] = np.nan
    return (drawn_auc * drawn_weight).sum(axis=1) / total


def run_bootstrap(config_path: str, artifact_dir: str, replicates: int, seed: int):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact_dir = Path(artifact_dir)
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    dataset = prepare_sequence_dataset(config["input_csv"], config)
    rng = np.random.default_rng(seed)
    result = {"replicates": replicates, "splits": {}}
    for split_name in ("validation", "test"):
        arrays = dataset.splits[split_name]
        users = arrays.raw_user_ids
        unique_users = np.unique(users)
        resample = rng.integers(0, len(unique_users), size=(replicates, len(unique_users)))
        observed = {}
        per_model_task_stats = {}
        for name, model_report in report["models"].items():
            spec = model_report["architecture"]
            model = FocusedRanker(
                architecture=spec["architecture"],
                task_layer=spec["task_layer"],
                num_users=dataset.vocabulary.num_users,
                num_items=dataset.vocabulary.num_items,
                embedding_dim=int(spec.get("embedding_dim", 32)),
                hidden_dim=int(spec.get("hidden_dim", 64)),
                num_experts=int(spec.get("num_experts", 4)),
            )
            import torch

            model.load_state_dict(torch.load(artifact_dir / f"{name}.pt", map_location="cpu"))
            predictions = _predict(model, arrays, 2048, torch.device("cpu"))
            task_scores = {}
            task_stats = {}
            for task in PRIMARY_TASKS:
                uid, aucs, weights = _per_user_auc(arrays.labels[task], predictions[task], users)
                assert (uid == unique_users).all()
                valid = weights > 0
                task_scores[task] = float(
                    (aucs[valid] * weights[valid]).sum() / weights[valid].sum()
                )
                task_stats[task] = (aucs, weights)
            observed[name] = {
                "is_click_gauc": task_scores["is_click"],
                "long_view_gauc": task_scores["long_view"],
                "selection_gauc": float(np.mean(list(task_scores.values()))),
            }
            per_model_task_stats[name] = task_stats
        selection_samples = {
            name: np.mean(
                [_selection_matrix(stats[task], resample) for task in PRIMARY_TASKS],
                axis=0,
            )
            for name, stats in per_model_task_stats.items()
        }
        pairs = {}
        names = list(report["models"])
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                delta = selection_samples[left] - selection_samples[right]
                delta = delta[~np.isnan(delta)]
                pairs[f"{left}_minus_{right}"] = {
                    "observed_delta": observed[left]["selection_gauc"]
                    - observed[right]["selection_gauc"],
                    "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
                    "p_delta_le_0": float((delta <= 0).mean()),
                }
        result["splits"][split_name] = {"models": observed, "pairs": pairs}
    return result


def collect_seeds(pattern: str):
    rows = []
    for path in sorted(glob.glob(pattern)):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        seeds_entry = {
            "report": path,
            "winner": report["winner"],
            "validation_selection_gauc": {
                name: model["validation_selection_gauc"] for name, model in report["models"].items()
            },
        }
        rows.append(seeds_entry)
    if not rows:
        return {"runs": [], "note": "no reports matched"}
    names = sorted(rows[0]["validation_selection_gauc"])
    summary = {}
    for name in names:
        values = [row["validation_selection_gauc"][name] for row in rows]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values": values,
        }
    winners = [row["winner"] for row in rows]
    return {
        "runs": rows,
        "summary": summary,
        "winner_counts": {name: winners.count(name) for name in set(winners)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sequence_ranking_real.json")
    parser.add_argument("--artifact-dir", default="artifacts/sequence-ranking-real")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument(
        "--seed-reports",
        default="",
        help="glob of per-seed report.json files to aggregate (optional)",
    )
    parser.add_argument("--output", default="artifacts/sequence-ranking-real/comparison.json")
    args = parser.parse_args()
    payload = {
        "method": {
            "bootstrap": "paired user-clustered bootstrap over evaluation users",
            "selection_metric": "mean impression-weighted user GAUC of is_click and long_view",
            "caveat": (
                "bootstrap quantifies evaluation-cohort sampling uncertainty for one "
                "training run; seed aggregation quantifies retraining variance"
            ),
        },
        "bootstrap": run_bootstrap(
            args.config, args.artifact_dir, args.replicates, args.bootstrap_seed
        ),
    }
    if args.seed_reports:
        payload["seeds"] = collect_seeds(args.seed_reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["bootstrap"]["splits"]["validation"]["pairs"], indent=2))
    if "seeds" in payload:
        print(json.dumps(payload["seeds"].get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
