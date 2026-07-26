FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY deployments ./deployments
COPY data/demo ./data/demo
COPY data/processed ./data/processed
COPY artifacts/benchmarks/kuairand-pure-sample.json ./artifacts/benchmarks/kuairand-pure-sample.json
COPY artifacts/sequence-ranking-real ./artifacts/sequence-ranking-real
COPY artifacts/transformer-comparison ./artifacts/transformer-comparison
COPY artifacts/serving ./artifacts/serving
COPY frontend ./frontend
RUN pip install --no-cache-dir '.[serving,streaming,ml]'

EXPOSE 8000
CMD ["uvicorn", "streamrank.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
