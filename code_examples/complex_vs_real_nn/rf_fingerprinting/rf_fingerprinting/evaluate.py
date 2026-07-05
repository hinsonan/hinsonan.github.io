"""Evaluation helpers for RF fingerprinting models."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits(model: torch.nn.Module, dataset, device: torch.device):
    """Collect logits and labels from a dataset.

    Args:
        model: Classifier model.
        dataset: Dataset yielding ``(iq, label)``.
        device: Compute device.

    Returns:
        Tuple ``(logits, labels)`` as NumPy arrays.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    logits = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        logits.append(out)
        labels.append(y.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute scalar metrics from logits and labels.

    Args:
        logits: Logits array ``[N, C]``.
        labels: True labels array ``[N]``.

    Returns:
        Dictionary with accuracy and macro F1.
    """
    pred = np.argmax(logits, axis=1)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
    }
