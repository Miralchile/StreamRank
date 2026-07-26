"""Evaluate the retrieval stage and the diversity rerank stage offline.

Retrieval: time-correct next-positive recall/hit-rate/coverage for ItemCF,
category-aware popularity and their RRF fusion (see evaluation.recall).

Rerank: on engine recommendations for sampled evaluated users, compare the
pure score-ordered top-K against the diversity-constrained top-K on category
concentration and retained score mass. Uses the bound deployment (requires
the ml extra for the focused ranker checkpoint).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from streamrank.data.loader import load_interactions
from streamrank.engine import RecommendationEngine
from streamrank.evaluation.recall import evaluate_recall
from streamrank.ranking.scoring import PolicyScorer
from streamrank.serving.manifest import DeploymentBundle

ROOT = Path(__file__).resolve().parents[1]


def iso_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def rerank_effect(events, fit_end_ms, queries_users, manifest_path, top_k, no_constraint_cap):
    bundle = DeploymentBundle.load(manifest_path)
    rerank_config = dict(bundle.components["rerank_policy"])
    scorer = PolicyScorer(rerank_config["score_weights"])
    engine = RecommendationEngine(
        bundle.manifest,
        scorer,
        model_config=bundle.components["model"],
        rerank_config=rerank_config,
    ).fit([event for event in events if event.event_time_ms < fit_end_ms])
    baseline_config = {
        "max_per_category": no_constraint_cap,
        "max_per_author": no_constraint_cap,
        "concentration_penalty": 0.0,
    }
    totals = {
        "users": 0,
        "baseline": {"categories": 0.0, "max_share": 0.0, "score": 0.0},
        "reranked": {"categories": 0.0, "max_share": 0.0, "score": 0.0},
        "overlap": 0.0,
    }
    for user_id in queries_users:
        outputs = {}
        for label, config in (("baseline", baseline_config), ("reranked", rerank_config)):
            engine.rerank_config = config
            outputs[label] = engine.recommend(user_id, top_k=top_k, query_time_ms=fit_end_ms, tab=0)
        if len(outputs["baseline"]) < top_k or len(outputs["reranked"]) < top_k:
            continue
        totals["users"] += 1
        for label, items in outputs.items():
            categories = [str(candidate.features.get("category")) for candidate in items]
            share = max(categories.count(value) for value in set(categories)) / len(items)
            totals[label]["categories"] += len(set(categories))
            totals[label]["max_share"] += share
            totals[label]["score"] += sum(candidate.rank_score for candidate in items)
        baseline_ids = {candidate.item_id for candidate in outputs["baseline"]}
        rerank_ids = {candidate.item_id for candidate in outputs["reranked"]}
        totals["overlap"] += len(baseline_ids & rerank_ids) / top_k
    users = max(1, totals["users"])
    return {
        "evaluated_users": totals["users"],
        "top_k": top_k,
        "baseline_score_order": {
            "mean_distinct_categories": round(totals["baseline"]["categories"] / users, 3),
            "mean_max_category_share": round(totals["baseline"]["max_share"] / users, 4),
        },
        "diversity_reranked": {
            "mean_distinct_categories": round(totals["reranked"]["categories"] / users, 3),
            "mean_max_category_share": round(totals["reranked"]["max_share"] / users, 4),
        },
        "score_retention": round(
            totals["reranked"]["score"] / max(1e-9, totals["baseline"]["score"]), 4
        ),
        "mean_topk_overlap": round(totals["overlap"] / users, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/processed/kuairand-pure-5k/interactions.csv")
    parser.add_argument("--fit-end", default="2022-05-01T00:00:00+08:00")
    parser.add_argument("--eval-end", default="2022-05-09T00:00:00+08:00")
    parser.add_argument("--manifest", default="deployments/manifests/kuairand-pure-sample.json")
    parser.add_argument("--rerank-users", type=int, default=200)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--output", default="artifacts/recall-eval/report.json")
    args = parser.parse_args()
    catalog_path = Path(args.catalog)
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    events = list(load_interactions(catalog_path))
    fit_end_ms = iso_to_ms(args.fit_end)
    eval_end_ms = iso_to_ms(args.eval_end)
    report = evaluate_recall(events, fit_end_ms, eval_end_ms)
    report["catalog"] = str(args.catalog)
    if not args.skip_rerank:
        from streamrank.evaluation.recall import build_queries

        queries = build_queries(events, fit_end_ms, eval_end_ms)
        sampled = [query.user_id for query in queries[: args.rerank_users]]
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        report["rerank_effect"] = rerank_effect(
            events, fit_end_ms, sampled, manifest_path, args.rerank_top_k, 10_000
        )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: report[k] for k in ("evaluated_users", "sources") if k in report}
    print(json.dumps(summary, indent=2))
    if "rerank_effect" in report:
        print(json.dumps(report["rerank_effect"], indent=2))


if __name__ == "__main__":
    main()
