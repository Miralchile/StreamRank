from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from streamrank.domain import Candidate

TASKS = ("is_click", "long_view", "is_like", "is_hate")


def _resolve_bound_path(value: object, artifact_dir: str | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if artifact_dir:
        candidate = (Path(artifact_dir) / path).resolve()
        if candidate.exists():
            return candidate
    return path.resolve()


def _verify_sha256(path: Path, expected: object | None) -> None:
    if not expected:
        return
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != str(expected):
        raise ValueError(f"bound ranker artifact digest mismatch: {path.name}")


class FocusedServingRanker:
    """Inference adapter for the offline-selected StreamRank focused ranker."""

    def __init__(
        self,
        model: object,
        *,
        user_vocabulary: Mapping[int, int],
        item_vocabulary: Mapping[int, int],
        numeric_mean: np.ndarray,
        numeric_std: np.ndarray,
        max_history: int,
    ):
        self.model = model.eval()
        self.user_vocabulary = dict(user_vocabulary)
        self.item_vocabulary = dict(item_vocabulary)
        self.numeric_mean = numeric_mean.astype(np.float32)
        self.numeric_std = numeric_std.astype(np.float32)
        self.numeric_std[self.numeric_std < 1e-6] = 1.0
        self.max_history = max_history

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "FocusedServingRanker":
        artifact_dir = str(config.get("_artifact_dir", ""))
        artifact_path = _resolve_bound_path(config["artifact_path"], artifact_dir)
        _verify_sha256(artifact_path, config.get("artifact_sha256"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        checkpoint_path = _resolve_bound_path(
            config.get("checkpoint_path") or artifact["checkpoint"],
            artifact_dir,
        )
        _verify_sha256(checkpoint_path, config.get("checkpoint_sha256"))
        model_spec = artifact["model"]
        from streamrank.focused.models import FocusedRanker

        model = FocusedRanker(
            architecture=str(model_spec["architecture"]),
            task_layer=str(model_spec["task_layer"]),
            num_users=int(artifact["num_users"]),
            num_items=int(artifact["num_items"]),
            embedding_dim=int(model_spec.get("embedding_dim", 32)),
            hidden_dim=int(model_spec.get("hidden_dim", 64)),
            num_experts=int(model_spec.get("num_experts", 4)),
        )
        import torch

        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        return cls(
            model,
            user_vocabulary={
                int(key): int(value) for key, value in artifact["user_vocabulary"].items()
            },
            item_vocabulary={
                int(key): int(value) for key, value in artifact["item_vocabulary"].items()
            },
            numeric_mean=np.asarray(artifact["numeric_mean"], dtype=np.float32),
            numeric_std=np.asarray(artifact["numeric_std"], dtype=np.float32),
            max_history=int(artifact["max_history"]),
        )

    def predict(
        self,
        *,
        user_id: int,
        history: Iterable[int],
        candidates: list[Candidate],
        query_time_ms: int,
        tab: int,
        user_stats: Mapping[int, tuple[int, int, int]],
        item_stats: Mapping[int, tuple[int, int, int]],
    ) -> list[dict[str, float]]:
        import torch

        mapped_history = [
            self.item_vocabulary.get(int(item_id), 1)
            for item_id in list(history)[-self.max_history :]
        ]
        padding = self.max_history - len(mapped_history)
        history_row = [0] * padding + mapped_history
        mask_row = [False] * padding + [True] * len(mapped_history)
        user_count, user_clicks, user_long = user_stats.get(user_id, (0, 0, 0))
        rows = []
        for candidate in candidates:
            item_count, item_clicks, item_long = item_stats.get(candidate.item_id, (0, 0, 0))
            upload_time_ms = int(candidate.features.get("upload_time_ms") or 0)
            age_days = (
                max(0.0, (query_time_ms - upload_time_ms) / 86_400_000) if upload_time_ms else 0.0
            )
            rows.append(
                [
                    math.log1p(user_count),
                    math.log1p(item_count),
                    user_clicks / max(1, user_count),
                    user_long / max(1, user_count),
                    item_clicks / max(1, item_count),
                    item_long / max(1, item_count),
                    tab / 14.0 if 0 <= tab <= 14 else -1.0,
                    math.log1p(age_days),
                ]
            )
        numeric = (np.asarray(rows, dtype=np.float32) - self.numeric_mean) / self.numeric_std
        batch = {
            "user_ids": torch.tensor(
                [self.user_vocabulary.get(user_id, 1)] * len(candidates), dtype=torch.long
            ),
            "item_ids": torch.tensor(
                [self.item_vocabulary.get(candidate.item_id, 1) for candidate in candidates],
                dtype=torch.long,
            ),
            "tab_ids": torch.tensor(
                [tab + 1 if 0 <= tab <= 14 else 0] * len(candidates), dtype=torch.long
            ),
            "histories": torch.tensor([history_row] * len(candidates), dtype=torch.long),
            "history_mask": torch.tensor([mask_row] * len(candidates), dtype=torch.bool),
            "numeric": torch.from_numpy(numeric),
        }
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = {
                task: torch.sigmoid(logits[task]).cpu().numpy().tolist() for task in TASKS
            }
        return [
            {task: float(probabilities[task][index]) for task in TASKS}
            for index in range(len(candidates))
        ]
