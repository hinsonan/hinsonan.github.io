"""Simple open-set scoring utilities for RF fingerprinting."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def max_softmax_scores(logits: np.ndarray) -> np.ndarray:
    """Compute max-softmax confidence scores.

    Args:
        logits: Logits array ``[N, C]``.

    Returns:
        Confidence scores in ``[0, 1]``.
    """
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    return np.max(prob, axis=1)


def open_set_auc(id_logits: np.ndarray, ood_logits: np.ndarray) -> Dict[str, float]:
    """Compute AUROC for in-distribution vs out-of-distribution detection.

    Args:
        id_logits: In-distribution logits.
        ood_logits: OOD logits.

    Returns:
        Dictionary with AUROC based on max-softmax confidence.
    """
    id_score = max_softmax_scores(id_logits)
    ood_score = max_softmax_scores(ood_logits)
    y = np.concatenate([np.ones_like(id_score), np.zeros_like(ood_score)])
    s = np.concatenate([id_score, ood_score])
    return {"open_set_auroc": float(roc_auc_score(y, s))}
