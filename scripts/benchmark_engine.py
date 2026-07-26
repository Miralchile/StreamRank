from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from streamrank.data.loader import load_interactions
from streamrank.engine import RecommendationEngine
from streamrank.ranking.scoring import PolicyScorer
from streamrank.serving.manifest import DeploymentBundle

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-qps", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=float("inf"))
    parser.add_argument("--catalog", default="data/demo/interactions.csv")
    parser.add_argument("--manifest", default="deployments/manifests/demo.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("requests and concurrency must be positive")

    catalog_path = Path(args.catalog)
    manifest_path = Path(args.manifest)
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    events = list(load_interactions(catalog_path))
    bundle = DeploymentBundle.load(manifest_path)
    scorer = PolicyScorer(bundle.components["rerank_policy"]["score_weights"])
    engine = RecommendationEngine(
        bundle.manifest,
        scorer,
        model_config=bundle.components["model"],
        rerank_config=bundle.components["rerank_policy"],
    ).fit(events)
    query_time = max(event.event_time_ms for event in events) + 1
    users = sorted({event.user_id for event in events})

    def request(index: int) -> float:
        started = time.perf_counter()
        engine.recommend(users[index % len(users)], top_k=10, query_time_ms=query_time)
        return (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    errors = 0
    latencies = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(request, index) for index in range(args.requests)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                errors += 1
    duration = time.perf_counter() - started
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "errors": errors,
        "duration_seconds": duration,
        "qps": (args.requests - errors) / max(duration, 1e-9),
        "p50_ms": percentile(latencies, 0.50) if latencies else None,
        "p95_ms": percentile(latencies, 0.95) if latencies else None,
        "p99_ms": percentile(latencies, 0.99) if latencies else None,
        "scope": "single-process local demo; not a production capacity claim",
        "catalog": str(catalog_path.relative_to(ROOT)),
        "deployment_id": bundle.manifest.deployment_id,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors or report["qps"] < args.min_qps:
        return 1
    if report["p95_ms"] is not None and report["p95_ms"] > args.max_p95_ms:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
