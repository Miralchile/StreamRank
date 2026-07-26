.PHONY: install install-platform preflight test lint audit serve serve-real docker verify rank-download rank-prepare rank-prepare-smoke rank-smoke rank-real build-deployment benchmark-real

install:
	python -m pip install -e '.[ml,dev]'

install-platform:
	python -m pip install -e '.[ml,serving,streaming,dev]'

# --- Single research track: KuaiRand sequence ranking ---------------------------------

rank-download:
	PYTHONPATH=src python -m streamrank.cli download-kuairand data/raw

rank-prepare:
	PYTHONPATH=src python -m streamrank.cli prepare-kuairand data/raw/KuaiRand-Pure/data data/processed/kuairand-pure-5k --users 5000 --seed 2026 --archive-md5 0820331067a3784d9691136f772b35a7

rank-prepare-smoke:
	PYTHONPATH=src python -m streamrank.cli prepare-kuairand data/raw/KuaiRand-Pure/data data/processed/kuairand-pure-sample --users 500 --seed 2026 --archive-md5 0820331067a3784d9691136f772b35a7

rank-smoke:
	PYTHONPATH=src python scripts/run_sequence_ranking.py --config configs/sequence_ranking_smoke.json

rank-real:
	PYTHONPATH=src python scripts/run_sequence_ranking.py --config configs/sequence_ranking_real.json

# --- Serving artifacts ----------------------------------------------------------------

build-deployment:
	PYTHONPATH=src python -m streamrank.cli build-deployment --root . --catalog data/processed/kuairand-pure-5k/interactions.csv --policy configs/serving_policy.json

benchmark-real:
	PYTHONPATH=src python scripts/benchmark_engine.py --requests 1000 --concurrency 8 --catalog data/processed/kuairand-pure-5k/interactions.csv --manifest deployments/manifests/kuairand-pure-sample.json --output artifacts/benchmarks/kuairand-pure-sample.json

audit:
	PYTHONPATH=src python -m streamrank.cli audit data/processed/kuairand-pure-5k/interactions.csv

# --- Quality gates --------------------------------------------------------------------

preflight:
	python -c 'import fastapi, redis; from kafka import KafkaProducer'

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

verify: lint test rank-smoke
	python -m compileall -q src scripts tests

# --- Serving --------------------------------------------------------------------------

serve:
	PYTHONPATH=src uvicorn streamrank.serving.app:app --reload

serve-real:
	STREAMRANK_MANIFEST=deployments/manifests/kuairand-pure-sample.json STREAMRANK_FALLBACK_MANIFEST=deployments/manifests/fallback.json STREAMRANK_DATASET_MANIFEST=data/processed/kuairand-pure-5k/dataset_manifest.json STREAMRANK_BENCHMARK_REPORT=artifacts/benchmarks/kuairand-pure-sample.json PYTHONPATH=src uvicorn streamrank.serving.app:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

compare-models:
	PYTHONPATH=src python scripts/compare_models.py --seed-reports 'artifacts/sequence-ranking-seeds/*/report.json'

recall-eval:
	PYTHONPATH=src python scripts/evaluate_recall.py

rank-transformers:
	PYTHONPATH=src python scripts/run_sequence_ranking.py --config configs/transformer_comparison.json
