"""Supervised fine-tuning for RF fingerprint classification."""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RFConfig
from .heads import ClassifierHead


class EncoderClassifier(nn.Module):
    """Wrap an encoder with a classifier head."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int, dropout: float = 0.0):
        """Initialize model.

        Args:
            encoder: Feature encoder.
            embed_dim: Embedding dimension.
            num_classes: Number of classes.
            dropout: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = encoder
        self.head = ClassifierHead(embed_dim, num_classes, dropout=dropout)

    def forward(self, x):
        """Compute class logits for a batch."""
        return self.head(self.encoder(x))


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy.

    Args:
        model: Classifier model.
        loader: Validation loader.
        device: Compute device.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
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


def _build_optimizer(model: nn.Module, cfg: RFConfig) -> torch.optim.Optimizer:
    """Build an optimizer with a lower LR for the encoder.

    Args:
        model: The encoder-classifier model.
        cfg: Runtime config.

    Returns:
        A configured Adam optimizer.
    """
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())

    param_groups = [
        {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_scale},
        {"params": head_params, "lr": cfg.lr},
    ]
    return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)


def finetune_classifier(
    encoder: nn.Module,
    train_dataset,
    val_dataset,
    num_classes: int,
    cfg: RFConfig,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, List[float]]]:
    """Fine-tune encoder + classifier head with supervised labels.

    Saves the model with the highest validation accuracy and returns it
    (not the final-epoch model).

    Args:
        encoder: Feature encoder.
        train_dataset: Supervised train dataset.
        val_dataset: Supervised validation dataset.
        num_classes: Number of device classes.
        cfg: Runtime config.
        device: Compute device.

    Returns:
        Tuple of ``(best_model, history)``.
    """
    model = EncoderClassifier(encoder, cfg.embed_dim, num_classes, dropout=cfg.dropout).to(device)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    opt = _build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()

    total_steps = cfg.finetune_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(total_steps, 1), eta_min=cfg.lr * 0.1
    )

    history: Dict[str, List[float]] = {"loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for epoch in range(cfg.finetune_epochs):
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
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            count += 1
        scheduler.step()

        val_acc = _accuracy(model, val_loader, device)
        history["loss"].append(running / max(count, 1))
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history