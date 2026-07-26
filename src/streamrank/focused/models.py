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


class SequenceTransformerEncoder(nn.Module):
    """Self-attention history encoders: SASRec-style (causal, candidate outside the
    attention) and BST-style (candidate participates as a sequence token).

    A learned lead token (BOS for sasrec, the candidate embedding for bst) occupies
    position zero, so every attention row has at least one valid key and empty
    histories cannot produce NaN attention outputs.
    """

    MAX_POSITIONS = 512

    def __init__(
        self,
        mode: str,
        num_users: int,
        num_items: int,
        numeric_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        if mode not in {"sasrec", "bst"}:
            raise ValueError(f"unknown sequence transformer mode: {mode}")
        self.mode = mode
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.tab_embedding = nn.Embedding(16, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.MAX_POSITIONS, embedding_dim)
        self.bos = nn.Parameter(torch.zeros(embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.Linear(6 * embedding_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        user = self.user_embedding(batch["user_ids"])
        item = self.item_embedding(batch["item_ids"])
        tab = self.tab_embedding(batch["tab_ids"])
        history = self.item_embedding(batch["histories"])
        batch_size, length, _ = history.shape
        lead = self.bos.expand(batch_size, 1, -1) if self.mode == "sasrec" else item.unsqueeze(1)
        sequence = torch.cat([lead, history], dim=1)
        valid = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=history.device),
                batch["history_mask"],
            ],
            dim=1,
        )
        positions = torch.arange(length + 1, device=history.device)
        sequence = sequence + self.position_embedding(positions)[None, :, :]
        causal = (
            torch.triu(
                torch.ones(length + 1, length + 1, dtype=torch.bool, device=history.device),
                diagonal=1,
            )
            if self.mode == "sasrec"
            else None
        )
        encoded = self.encoder(sequence, mask=causal, src_key_padding_mask=~valid)
        if self.mode == "sasrec":
            last_index = valid.sum(dim=1) - 1
            interest = encoded[torch.arange(batch_size, device=history.device), last_index]
        else:
            interest = encoded[:, 0]
        blocks = [user, item, tab, user * item, interest, interest * item]
        return self.projection(torch.cat([*blocks, batch["numeric"]], dim=-1))


class FieldAttentionEncoder(nn.Module):
    """AutoInt-style multi-head self-attention over feature fields (no sequence
    attention): user, candidate item, tab, projected numerics and mean-pooled
    history interact as five field tokens."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        numeric_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.tab_embedding = nn.Embedding(16, embedding_dim, padding_idx=0)
        self.numeric_projection = nn.Linear(numeric_dim, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.Linear(5 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        history = self.item_embedding(batch["histories"])
        mask = batch["history_mask"].unsqueeze(-1).float()
        pooled = (history * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        fields = torch.stack(
            [
                self.user_embedding(batch["user_ids"]),
                self.item_embedding(batch["item_ids"]),
                self.tab_embedding(batch["tab_ids"]),
                self.numeric_projection(batch["numeric"]),
                pooled,
            ],
            dim=1,
        )
        encoded = self.encoder(fields)
        return self.projection(encoded.flatten(start_dim=1))


class FocusedRanker(nn.Module):
    """One protocol, five encoder families, matched shared-bottom/MMoE decoders."""

    ARCHITECTURES = ("deepfm", "din", "sasrec", "bst", "autoint")

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
        num_heads: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        if task_layer not in {"shared_bottom", "mmoe"}:
            raise ValueError(f"unknown task layer: {task_layer}")
        self.architecture = architecture
        self.task_layer = task_layer
        if architecture in {"sasrec", "bst"}:
            self.encoder: nn.Module = SequenceTransformerEncoder(
                architecture,
                num_users,
                num_items,
                numeric_dim,
                embedding_dim,
                hidden_dim,
                num_heads,
                num_layers,
            )
        elif architecture == "autoint":
            self.encoder = FieldAttentionEncoder(
                num_users,
                num_items,
                numeric_dim,
                embedding_dim,
                hidden_dim,
                num_heads,
                num_layers,
            )
        else:
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
