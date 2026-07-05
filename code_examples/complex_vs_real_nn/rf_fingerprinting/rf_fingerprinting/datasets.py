"""Torch dataset wrappers for RF fingerprinting tasks."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import RFConfig
from .transforms import augment_iq


def split_indices(n: int, test_size: float, val_size: float, seed: int) -> Dict[str, np.ndarray]:
    """Create train/val/test index splits.

    Args:
        n: Number of samples.
        test_size: Fraction for test split.
        val_size: Fraction for validation split from train partition.
        seed: RNG seed.

    Returns:
        Dictionary with ``train``, ``val``, and ``test`` index arrays.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    rem = idx[n_test:]
    n_val = int(rem.shape[0] * val_size)
    val_idx = rem[:n_val]
    train_idx = rem[n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


class IQDataset(Dataset):
    """Supervised dataset for RF device classification.

    Returns normalized complex IQ tensors and integer device labels.
    """

    def __init__(self, iq: np.ndarray, device_id: np.ndarray):
        """Initialize dataset.

        Args:
            iq: Complex waveforms with shape ``[N, T]``.
            device_id: Device labels with shape ``[N]``.
        """
        self.iq = iq.astype(np.complex64)
        self.device_id = device_id.astype(np.int64)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.iq.shape[0]

    def __getitem__(self, index: int):
        """Fetch one supervised sample.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(iq_tensor, label_tensor)``.
        """
        x = torch.from_numpy(self.iq[index])
        y = torch.tensor(self.device_id[index], dtype=torch.long)
        return x, y


class TwoViewIQDataset(Dataset):
    """Contrastive dataset that yields two augmented views per sample."""

    def __init__(self, iq: np.ndarray, cfg: RFConfig):
        """Initialize two-view dataset.

        Args:
            iq: Complex waveforms with shape ``[N, T]``.
            cfg: Runtime config with augmentation settings.
        """
        self.iq = iq.astype(np.complex64)
        self.cfg = cfg

    def __len__(self) -> int:
        """Return number of samples."""
        return self.iq.shape[0]

    def __getitem__(self, index: int):
        """Fetch one contrastive sample with two views.

        Args:
            index: Sample index.

        Returns:
            Tuple ``(view1_tensor, view2_tensor)``.
        """
        base = self.iq[index]
        rng = np.random.default_rng()
        v1 = augment_iq(
            base,
            noise_std=self.cfg.noise_std,
            phase_jitter_rad=self.cfg.phase_jitter_rad,
            time_shift=self.cfg.time_shift,
            rng=rng,
        )
        v2 = augment_iq(
            base,
            noise_std=self.cfg.noise_std,
            phase_jitter_rad=self.cfg.phase_jitter_rad,
            time_shift=self.cfg.time_shift,
            rng=rng,
        )
        return torch.from_numpy(v1), torch.from_numpy(v2)
