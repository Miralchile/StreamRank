from __future__ import annotations

import torch
from torch import Tensor, nn

from streamrank.focused.dataset import TASKS
from streamrank.ranking.torch_models import DINAttention, MMoE


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        architecture: str,
        num_users: int,
        num_items: int,
        numeric_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.architecture = architecture
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.tab_embedding = nn.Embedding(16, embedding_dim, padding_idx=0)
        self.attention = DINAttention(embedding_dim) if architecture == "din" else None
        # user, item, tab, pairwise item-user and optional DIN interest/item-interest.
        blocks = 4 if architecture == "deepfm" else 6
        self.projection = nn.Sequential(
            nn.Linear(blocks * embedding_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        user = self.user_embedding(batch["user_ids"])
        item = self.item_embedding(batch["item_ids"])
        tab = self.tab_embedding(batch["tab_ids"])
        blocks = [user, item, tab, user * item]
        if self.attention is not None:
            history = self.item_embedding(batch["histories"])
            interest = self.attention(item, history, batch["history_mask"])
            blocks.extend([interest, interest * item])
        return self.projection(torch.cat([*blocks, batch["numeric"]], dim=-1))


class FocusedRanker(nn.Module):
    """One encoder family with matched shared-bottom/MMoE task decoders."""

    def __init__(
        self,
        architecture: str,
        task_layer: str,
        num_users: int,
        num_items: int,
        numeric_dim: int = 8,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_experts: int = 4,
    ):
        super().__init__()
        if architecture not in {"deepfm", "din"}:
            raise ValueError(f"unknown architecture: {architecture}")
        if task_layer not in {"shared_bottom", "mmoe"}:
            raise ValueError(f"unknown task layer: {task_layer}")
        self.architecture = architecture
        self.task_layer = task_layer
        self.encoder = FeatureEncoder(
            architecture, num_users, num_items, numeric_dim, embedding_dim, hidden_dim
        )
        if task_layer == "mmoe":
            self.decoder: nn.Module = MMoE(hidden_dim, hidden_dim, TASKS, num_experts)
        else:
            self.decoder = nn.ModuleDict({task: nn.Linear(hidden_dim, 1) for task in TASKS})

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        shared = self.encoder(batch)
        if self.task_layer == "mmoe":
            return self.decoder(shared)  # type: ignore[operator, no-any-return]
        return {
            task: head(shared).squeeze(-1)
            for task, head in self.decoder.items()  # type: ignore[union-attr]
        }


def weighted_multitask_loss(
    logits: dict[str, Tensor], batch: dict[str, Tensor], task_weights: dict[str, float]
) -> Tensor:
    losses = []
    for task, output in logits.items():
        weight = float(task_weights.get(task, 0.0))
        if weight:
            losses.append(
                weight * nn.functional.binary_cross_entropy_with_logits(output, batch[task])
            )
    if not losses:
        raise ValueError("at least one task weight must be positive")
    return torch.stack(losses).sum() / sum(task_weights.get(task, 0.0) for task in logits)
