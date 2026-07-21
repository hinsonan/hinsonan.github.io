"""Evaluation helpers for emitter identification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn


def predictions(model: nn.Module, iq: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict emitter labels."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(iq).to(device)).argmax(1).cpu().numpy()


def emitter_metrics(labels: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return confusion matrix and per-emitter accuracy."""
    matrix = confusion_matrix(labels, predicted)
    return matrix, matrix.diagonal() / matrix.sum(axis=1)


def open_set_auroc(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    known_embeddings: np.ndarray,
    unknown_embeddings: np.ndarray,
) -> float:
    """Score unknown emitters using cosine similarity to known prototypes."""
    normalize = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    train = normalize(train_embeddings)
    prototypes = np.stack([train[train_labels == label].mean(0) for label in np.unique(train_labels)])
    prototypes = normalize(prototypes)
    known_scores = normalize(known_embeddings) @ prototypes.T
    unknown_scores = normalize(unknown_embeddings) @ prototypes.T
    scores = np.concatenate((known_scores.max(1), unknown_scores.max(1)))
    targets = np.concatenate((np.ones(len(known_scores)), np.zeros(len(unknown_scores))))
    return roc_auc_score(targets, scores)
