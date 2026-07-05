"""Visualization utilities for RF fingerprinting notebooks and scripts."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_iq_views(base: np.ndarray, views: Iterable[np.ndarray], title: str = "Augmentation Preview"):
    """Plot original IQ trace and augmented views.

    Args:
        base: Original waveform ``[T]``.
        views: Iterable of augmented waveforms.
        title: Figure title.
    """
    views = list(views)
    fig, axes = plt.subplots(1, 1 + len(views), figsize=(3.6 * (1 + len(views)), 3.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    all_series = [base] + views
    names = ["base"] + [f"view_{i+1}" for i in range(len(views))]
    for ax, sig, name in zip(axes, all_series, names):
        ax.plot(sig.real, label="I", alpha=0.9)
        ax.plot(sig.imag, label="Q", alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("sample")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_history(history: dict, key: str, title: str):
    """Plot one scalar training history curve.

    Args:
        history: Dictionary containing list values.
        key: Metric key to plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(history.get(key, []), marker="o")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
