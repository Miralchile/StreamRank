from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from streamrank.evaluation.metrics import binary_classification_report, gauc
from streamrank.focused.dataset import (
    TASKS,
    PreparedDataset,
    TorchSequenceDataset,
    prepare_sequence_dataset,
)
from streamrank.focused.models import FocusedRanker, weighted_multitask_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _predict(
    model: FocusedRanker, arrays, batch_size: int, device: torch.device
) -> dict[str, np.ndarray]:
    loader = DataLoader(TorchSequenceDataset(arrays), batch_size=batch_size, shuffle=False)
    output = {task: [] for task in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(_to_device(batch, device))
            for task in TASKS:
                output[task].append(torch.sigmoid(logits[task]).cpu().numpy())
    return {task: np.concatenate(parts) for task, parts in output.items()}


def _metrics(arrays, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for task in TASKS:
        labels = arrays.labels[task].astype(int).tolist()
        scores = predictions[task].tolist()
        task_report = binary_classification_report(labels, scores)
        task_report.update(gauc(labels, scores, arrays.raw_user_ids.tolist()))
        task_report["positive_rows"] = int(arrays.labels[task].sum())
        report[task] = task_report
    return report


def _selection_score(metrics: dict[str, Any]) -> float:
    # Primary objectives only. Sparse like/hate tasks never decide early stopping.
    values = [metrics[task].get("gauc") for task in ("is_click", "long_view")]
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return sum(valid) / len(valid) if valid else float("-inf")


def _json_safe(value: Any) -> Any:
    """Make undefined rare-label metrics explicit instead of emitting non-standard NaN JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _train_one(
    name: str,
    spec: dict[str, Any],
    dataset: PreparedDataset,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    train_config = config["training"]
    device = torch.device(train_config.get("device", "cpu"))
    model = FocusedRanker(
        architecture=spec["architecture"],
        task_layer=spec["task_layer"],
        num_users=dataset.vocabulary.num_users,
        num_items=dataset.vocabulary.num_items,
        embedding_dim=int(spec.get("embedding_dim", 32)),
        hidden_dim=int(spec.get("hidden_dim", 64)),
        num_experts=int(spec.get("num_experts", 4)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-5)),
    )
    loader = DataLoader(
        TorchSequenceDataset(dataset.splits["train"]),
        batch_size=int(train_config.get("batch_size", 512)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.get("seed", 2026))),
    )
    task_weights = {key: float(value) for key, value in train_config["task_weights"].items()}
    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    patience = int(train_config.get("patience", 2))
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(train_config.get("epochs", 5)) + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = _to_device(batch, device)
            loss = weighted_multitask_loss(model(batch), batch, task_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = _predict(
            model, dataset.splits["validation"], int(train_config.get("batch_size", 512)), device
        )
        validation_metrics = _metrics(dataset.splits["validation"], validation_predictions)
        score = _selection_score(validation_metrics)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_gauc": score}
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model_path = output_dir / f"{name}.pt"
    torch.save(best_state, model_path)
    split_metrics = {}
    for split_name in ("validation", "test", "random_diagnostic"):
        if split_name in dataset.splits and len(dataset.splits[split_name]):
            predictions = _predict(
                model, dataset.splits[split_name], int(train_config.get("batch_size", 512)), device
            )
            split_metrics[split_name] = _metrics(dataset.splits[split_name], predictions)
    return {
        "architecture": spec,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "validation_selection_gauc": best_score,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "metrics": split_metrics,
        "checkpoint": str(model_path),
    }


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _seed_everything(int(config.get("seed", 2026)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = prepare_sequence_dataset(config["input_csv"], config)
    results = {}
    for name, spec in config["models"].items():
        results[name] = _train_one(name, spec, dataset, config, output_dir)
    winner = max(results, key=lambda name: results[name]["validation_selection_gauc"])
    artifact = _json_safe(
        {
            "project_scope": "multi-objective sequential ranking",
            "primary_tasks": ["is_click", "long_view"],
            "auxiliary_task": "is_like",
            "diagnostic_task": "is_hate",
            "selection_rule": "mean validation user-GAUC of is_click and long_view",
            "dataset": dataset.metadata,
            "models": results,
            "winner": winner,
            "limitations": [
                "KuaiRand-Pure histories are limited to candidate-pool interactions.",
                "Random exposure results are a bias diagnostic, not arbitrary-policy OPE.",
                "Public logged data cannot establish live online business uplift.",
            ],
        }
    )
    (output_dir / "report.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "artifact.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "model": results[winner]["architecture"],
                "checkpoint": results[winner]["checkpoint"],
                "num_users": dataset.vocabulary.num_users,
                "num_items": dataset.vocabulary.num_items,
                "numeric_mean": dataset.numeric_mean.tolist(),
                "numeric_std": dataset.numeric_std.tolist(),
                "user_vocabulary": {
                    str(key): value for key, value in dataset.vocabulary.users.items()
                },
                "item_vocabulary": {
                    str(key): value for key, value in dataset.vocabulary.items.items()
                },
                "max_history": dataset.metadata["max_history"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact
