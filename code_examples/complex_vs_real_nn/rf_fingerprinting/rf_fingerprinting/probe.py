"""Linear probing utilities for RF embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(encoder: torch.nn.Module, dataset, device: torch.device):
    """Extract embeddings and labels from a supervised dataset.

    Args:
        encoder: Feature encoder.
        dataset: Dataset yielding ``(iq, label)``.
        device: Compute device.

    Returns:
        Tuple ``(embeddings, labels)`` as NumPy arrays.
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x).cpu().numpy()
        feats.append(z)
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run_linear_probe(
    encoder: torch.nn.Module,
    train_dataset,
    test_dataset,
    device: torch.device,
    max_iter: int = 300,
) -> Dict[str, float]:
    """Train and evaluate a linear probe on frozen embeddings.

    Args:
        encoder: Frozen feature encoder.
        train_dataset: Supervised train dataset.
        test_dataset: Supervised test dataset.
        device: Compute device.
        max_iter: Max logistic regression iterations.

    Returns:
        Metrics dictionary.
    """
    x_train, y_train = extract_embeddings(encoder, train_dataset, device)
    x_test, y_test = extract_embeddings(encoder, test_dataset, device)
    clf = LogisticRegression(max_iter=max_iter, n_jobs=1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {"probe_acc": float(accuracy_score(y_test, pred))}
