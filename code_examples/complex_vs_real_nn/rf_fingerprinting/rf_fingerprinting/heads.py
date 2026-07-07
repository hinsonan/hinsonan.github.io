"""Task heads for RF fingerprinting models."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head used in SimCLR pretraining."""

    def __init__(self, in_dim: int, proj_dim: int = 64):
        """Initialize projection head.

        Args:
            in_dim: Input embedding dimension.
            proj_dim: Projection output dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        """Project embeddings for contrastive loss."""
        return self.net(x)


class ClassifierHead(nn.Module):
    """Linear classifier head for probing and fine-tuning."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        """Initialize classifier head.

        Args:
            in_dim: Input embedding dimension.
            num_classes: Number of classes.
            dropout: Dropout probability before the linear layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """Compute logits from embeddings."""
        return self.fc(self.dropout(x))
