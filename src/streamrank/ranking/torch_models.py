from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without the ml extra
    raise ImportError("Install StreamRank with the 'ml' extra to use torch models") from exc


class TwoTowerModel(nn.Module):
    """ID + dense/content feature towers; content inputs permit controlled cold-item tests."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_dense_dim: int,
        item_dense_dim: int,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.user_id = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_id = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_dense = nn.Linear(user_dense_dim, embedding_dim)
        self.item_dense = nn.Linear(item_dense_dim, embedding_dim)
        self.user_projection = nn.Sequential(nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.item_projection = nn.Sequential(nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))

    def encode_user(self, user_ids: Tensor, user_dense: Tensor) -> Tensor:
        return F.normalize(
            self.user_projection(self.user_id(user_ids) + self.user_dense(user_dense)),
            dim=-1,
        )

    def encode_item(self, item_ids: Tensor, item_dense: Tensor) -> Tensor:
        return F.normalize(
            self.item_projection(self.item_id(item_ids) + self.item_dense(item_dense)),
            dim=-1,
        )

    def forward(
        self, user_ids: Tensor, item_ids: Tensor, user_dense: Tensor, item_dense: Tensor
    ) -> Tensor:
        return (
            self.encode_user(user_ids, user_dense) * self.encode_item(item_ids, item_dense)
        ).sum(-1)


class DINAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, candidate: Tensor, history: Tensor, mask: Tensor) -> Tensor:
        expanded = candidate.unsqueeze(1).expand_as(history)
        inputs = torch.cat([expanded, history, expanded - history, expanded * history], dim=-1)
        logits = self.network(inputs).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return (weights.unsqueeze(-1) * history).sum(dim=1)


class MMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        tasks: tuple[str, ...],
        num_experts: int = 4,
    ):
        super().__init__()
        self.tasks = tasks
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, expert_dim), nn.ReLU()) for _ in range(num_experts)]
        )
        self.gates = nn.ModuleDict({task: nn.Linear(input_dim, num_experts) for task in tasks})
        self.heads = nn.ModuleDict({task: nn.Linear(expert_dim, 1) for task in tasks})

    def forward(self, shared: Tensor) -> dict[str, Tensor]:
        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        outputs = {}
        for task in self.tasks:
            gate = torch.softmax(self.gates[task](shared), dim=-1).unsqueeze(-1)
            mixed = (expert_outputs * gate).sum(dim=1)
            outputs[task] = self.heads[task](mixed).squeeze(-1)
        return outputs


class DINMMoERanker(nn.Module):
    TASKS = ("is_click", "long_view", "is_like", "is_hate")

    def __init__(
        self,
        num_items: int,
        numeric_dim: int,
        embedding_dim: int = 32,
        expert_dim: int = 64,
        num_experts: int = 4,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.attention = DINAttention(embedding_dim)
        shared_dim = embedding_dim * 3 + numeric_dim
        self.mmoe = MMoE(shared_dim, expert_dim, self.TASKS, num_experts)

    def forward(
        self,
        item_ids: Tensor,
        history_item_ids: Tensor,
        history_mask: Tensor,
        numeric_features: Tensor,
    ) -> dict[str, Tensor]:
        candidate = self.item_embedding(item_ids)
        history = self.item_embedding(history_item_ids)
        interest = self.attention(candidate, history, history_mask)
        shared = torch.cat([candidate, interest, candidate * interest, numeric_features], dim=-1)
        return self.mmoe(shared)


def multitask_loss(
    logits: dict[str, Tensor],
    labels: dict[str, Tensor],
    task_weights: dict[str, float] | None = None,
    consistency_scene_mask: Tensor | None = None,
    consistency_weight: float = 0.0,
) -> Tensor:
    weights = task_weights or {task: 1.0 for task in logits}
    loss = sum(
        weights.get(task, 1.0) * F.binary_cross_entropy_with_logits(value, labels[task].float())
        for task, value in logits.items()
    )
    if consistency_scene_mask is not None and consistency_weight > 0:
        click = torch.sigmoid(logits["is_click"])
        long_view = torch.sigmoid(logits["long_view"])
        violations = F.relu(long_view - click) * consistency_scene_mask.float()
        denominator = consistency_scene_mask.float().sum().clamp_min(1.0)
        loss = loss + consistency_weight * violations.sum() / denominator
    return loss
