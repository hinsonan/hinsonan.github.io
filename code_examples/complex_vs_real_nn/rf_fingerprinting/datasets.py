"""Dataset classes, splits, and RF IQ augmentations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import RFConfig


def split_indices(
    n: int,
    test_size: float,
    val_size: float,
    seed: int,
    labels: np.ndarray | None = None,
    session_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Create train/val/test index splits.

    If ``session_ids`` is provided, splits are made **by session**: test
    sessions are held out entirely so that evaluation measures generalization
    to unseen channel conditions. If only ``labels`` is given, splits are
    stratified by device class.

    Args:
        n: Number of samples.
        test_size: Fraction for test split.
        val_size: Fraction for validation split from train partition.
        seed: RNG seed.
        labels: Optional class labels for stratified splitting.
        session_ids: Optional session labels for session-held-out splitting.

    Returns:
        Dictionary with ``train``, ``val``, and ``test`` index arrays.
    """
    rng = np.random.default_rng(seed)

    if session_ids is not None:
        session_ids = np.asarray(session_ids)
        unique_sessions = np.unique(session_ids)
        rng.shuffle(unique_sessions)
        n_test_sessions = max(1, int(len(unique_sessions) * test_size))
        n_val_sessions = max(1, int(len(unique_sessions) * val_size))
        test_sessions = set(unique_sessions[:n_test_sessions].tolist())
        val_sessions = set(
            unique_sessions[n_test_sessions : n_test_sessions + n_val_sessions].tolist()
        )
        test_mask = np.array([sid in test_sessions for sid in session_ids], dtype=bool)
        val_mask = np.array([sid in val_sessions for sid in session_ids], dtype=bool)
        train_mask = ~(test_mask | val_mask)
        return {
            "train": np.flatnonzero(train_mask),
            "val": np.flatnonzero(val_mask),
            "test": np.flatnonzero(test_mask),
        }

    if labels is None:
        idx = rng.permutation(n)
        n_test = int(n * test_size)
        test_idx = idx[:n_test]
        rem = idx[n_test:]
        n_val = int(rem.shape[0] * val_size)
        val_idx = rem[:n_val]
        train_idx = rem[n_val:]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    labels = np.asarray(labels)
    train_parts = []
    val_parts = []
    test_parts = []
    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        cls_idx = rng.permutation(cls_idx)
        n_test = int(cls_idx.shape[0] * test_size)
        cls_test = cls_idx[:n_test]
        cls_rem = cls_idx[n_test:]
        n_val = int(cls_rem.shape[0] * val_size)
        cls_val = cls_rem[:n_val]
        cls_train = cls_rem[n_val:]
        test_parts.append(cls_test)
        val_parts.append(cls_val)
        train_parts.append(cls_train)

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def apply_phase_jitter(
    iq: np.ndarray, max_jitter_rad: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a random global phase offset.

    Args:
        iq: Complex waveform of shape ``[T]``.
        max_jitter_rad: Max absolute phase offset in radians.
        rng: NumPy random generator.

    Returns:
        Phase-jittered waveform.
    """
    phi = rng.uniform(-max_jitter_rad, max_jitter_rad)
    return (iq * np.exp(1j * phi)).astype(np.complex64)


def apply_awgn(iq: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex white Gaussian noise.

    Args:
        iq: Complex waveform.
        noise_std: Standard deviation per real/imag component.
        rng: NumPy random generator.

    Returns:
        Noisy waveform.
    """
    noise = noise_std * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def apply_time_shift(iq: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a random circular time shift.

    Args:
        iq: Complex waveform.
        max_shift: Max absolute shift samples.
        rng: NumPy random generator.

    Returns:
        Shifted waveform.
    """
    if max_shift <= 0:
        return iq
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(iq, shift).astype(np.complex64)


def apply_cfo(
    iq: np.ndarray, max_cfo_per_sample: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a random carrier frequency offset.

    Args:
        iq: Complex waveform.
        max_cfo_per_sample: Max absolute frequency offset per sample (radians).
        rng: NumPy random generator.

    Returns:
        Waveform with CFO applied.
    """
    cfo = rng.uniform(-max_cfo_per_sample, max_cfo_per_sample)
    t = np.arange(iq.shape[0], dtype=np.float32)
    return (iq * np.exp(1j * cfo * t)).astype(np.complex64)


def apply_amplitude_jitter(
    iq: np.ndarray, max_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a random global amplitude scale.

    Args:
        iq: Complex waveform.
        max_scale: Max relative amplitude perturbation.
        rng: NumPy random generator.

    Returns:
        Amplitude-scaled waveform.
    """
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    return (iq * scale).astype(np.complex64)


def normalize_power(iq: np.ndarray) -> np.ndarray:
    """Normalize waveform to unit average power.

    Args:
        iq: Complex waveform.

    Returns:
        Unit-power waveform.
    """
    p = np.sqrt(np.mean(np.abs(iq) ** 2) + 1e-8)
    return (iq / p).astype(np.complex64)


def augment_iq(
    iq: np.ndarray,
    noise_std: float = 0.04,
    phase_jitter_rad: float = 0.05,
    time_shift: int = 8,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    cfo_jitter_rad: float = 0.0,
    amplitude_jitter: float = 0.0,
    aug_prob: float = 1.0,
) -> np.ndarray:
    """Create one augmented view of an IQ waveform.

    Each transform is applied independently with probability ``aug_prob``.
    Conservative defaults are used so that device fingerprints are not
    erased during SimCLR pretraining.

    Args:
        iq: Complex waveform of shape ``[T]``.
        noise_std: Noise standard deviation.
        phase_jitter_rad: Phase jitter range.
        time_shift: Max circular shift in samples.
        seed: Optional seed if ``rng`` is not provided.
        rng: Optional pre-built random generator.
        cfo_jitter_rad: Max CFO per sample in radians.
        amplitude_jitter: Max relative amplitude perturbation.
        aug_prob: Probability of applying each stochastic transform.

    Returns:
        Augmented waveform.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    out = iq.astype(np.complex64, copy=True)

    if rng.random() < aug_prob:
        out = apply_phase_jitter(out, phase_jitter_rad, rng)
    if rng.random() < aug_prob and time_shift > 0:
        out = apply_time_shift(out, time_shift, rng)
    if rng.random() < aug_prob and cfo_jitter_rad > 0:
        out = apply_cfo(out, cfo_jitter_rad, rng)
    if rng.random() < aug_prob and amplitude_jitter > 0:
        out = apply_amplitude_jitter(out, amplitude_jitter, rng)
    if rng.random() < aug_prob:
        out = apply_awgn(out, noise_std, rng)

    return normalize_power(out)


class IQDataset(Dataset):
    """Supervised dataset for RF device classification."""

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
        aug_kwargs = dict(
            noise_std=self.cfg.noise_std,
            phase_jitter_rad=self.cfg.phase_jitter_rad,
            time_shift=self.cfg.time_shift,
            cfo_jitter_rad=self.cfg.cfo_jitter_rad,
            amplitude_jitter=self.cfg.amplitude_jitter,
            aug_prob=self.cfg.aug_prob,
            rng=rng,
        )
        v1 = augment_iq(base, **aug_kwargs)
        v2 = augment_iq(base, **aug_kwargs)
        return torch.from_numpy(v1), torch.from_numpy(v2)
