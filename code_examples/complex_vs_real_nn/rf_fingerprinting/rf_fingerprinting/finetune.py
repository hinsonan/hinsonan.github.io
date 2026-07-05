"""Supervised fine-tuning for RF fingerprint classification."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RFConfig
from .heads import ClassifierHead


class EncoderClassifier(nn.Module):
    """Wrap an encoder with a classifier head."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int):
        """Initialize model.

        Args:
            encoder: Feature encoder.
            embed_dim: Embedding dimension.
            num_classes: Number of classes.
        """
        super().__init__()
        self.encoder = encoder
        self.head = ClassifierHead(embed_dim, num_classes)

    def forward(self, x):
        """Compute class logits for a batch."""
        return self.head(self.encoder(x))


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy."""
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += int((pred == y).sum().item())
    return correct / max(total, 1)


def finetune_classifier(
    encoder: nn.Module,
    train_dataset,
    val_dataset,
    num_classes: int,
    cfg: RFConfig,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, List[float]]]:
    """Fine-tune encoder + classifier head with supervised labels.

    Args:
        encoder: Feature encoder.
        train_dataset: Supervised train dataset.
        val_dataset: Supervised validation dataset.
        num_classes: Number of device classes.
        cfg: Runtime config.
        device: Compute device.

    Returns:
        Tuple of ``(trained_model, history)``.
    """
    model = EncoderClassifier(encoder, cfg.embed_dim, num_classes).to(device)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "val_acc": []}
    for _ in range(cfg.finetune_epochs):
        model.train()
        running = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1
        history["loss"].append(running / max(count, 1))
        history["val_acc"].append(_accuracy(model, val_loader, device))
    return model, history
