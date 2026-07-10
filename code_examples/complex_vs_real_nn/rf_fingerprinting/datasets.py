"""Dataset classes, splits, and RF IQ augmentations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .config import RFConfig
except ImportError:
    from config import RFConfig


def split_indices(
    n: int,
    test_size: float,
    val_size: float,
    seed: int,
    labels: np.ndarray | None = None,
    session_ids: np.ndarray | None = None,
    device_ids: np.ndarray | None = None,
    receiver_ids: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    group_by: str | None = None,
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

    if n <= 0 or not 0.0 <= test_size <= 1.0 or not 0.0 <= val_size <= 1.0:
        raise ValueError("n must be positive and split sizes must be in [0, 1].")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    if group_by not in (None, "session", "device", "receiver", "combined"):
        raise ValueError("group_by must be session, device, receiver, combined, or None.")
    required = {"session": session_ids, "device": device_ids, "receiver": receiver_ids}
    if group_by in required and required[group_by] is None:
        raise ValueError(f"group_by='{group_by}' requires matching metadata.")
    metadata = [x for x in (session_ids, device_ids, receiver_ids) if x is not None]
    if metadata:
        if any(np.asarray(x).ndim != 1 or len(x) != n for x in metadata):
            raise ValueError("All grouping metadata must be one-dimensional and have length n.")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != n:
            raise ValueError("labels must be one-dimensional and have length n.")
        if labels.dtype.kind == "f" and not np.all(np.isfinite(labels)):
            raise ValueError("labels must contain finite values.")
        if labels.size == 0:
            raise ValueError("labels must not be empty.")
    if group_by is not None:
        required = {"session": session_ids, "device": device_ids, "receiver": receiver_ids}.get(group_by)
        if group_by != "combined" and required is None:
            raise ValueError(f"group_by='{group_by}' requires matching metadata.")
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.ndim != 1 or len(group_ids) != n:
            raise ValueError("group_ids must be one-dimensional and have length n.")
        selected_groups = group_ids
    elif group_by == "combined":
        if not metadata:
            raise ValueError("combined grouping requires session, device, or receiver metadata.")
        selected_groups = np.asarray(list(zip(*(np.asarray(x).tolist() for x in metadata))), dtype=str)
    elif group_by == "device" and device_ids is not None:
        selected_groups = np.asarray(device_ids)
    elif group_by == "receiver" and receiver_ids is not None:
        selected_groups = np.asarray(receiver_ids)
    elif session_ids is not None:
        selected_groups = np.asarray(session_ids)
    elif device_ids is not None:
        selected_groups = np.asarray(device_ids)
    elif receiver_ids is not None:
        selected_groups = np.asarray(receiver_ids)
    else:
        selected_groups = None

    if selected_groups is not None:
        unique_groups, inverse = np.unique(selected_groups, axis=0, return_inverse=True) if selected_groups.ndim > 1 else np.unique(selected_groups, return_inverse=True)
        order = rng.permutation(len(unique_groups))
        n_test_groups = max(1, int(len(order) * test_size)) if test_size > 0 else 0
        remaining = len(order) - n_test_groups
        n_val_groups = max(1, int(remaining * val_size)) if val_size > 0 and remaining else 0
        test_groups = set(order[:n_test_groups].tolist())
        val_groups = set(order[n_test_groups:n_test_groups + n_val_groups].tolist())
        test_mask = np.isin(inverse, list(test_groups))
        val_mask = np.isin(inverse, list(val_groups))
        result = {
            "train": np.flatnonzero(~(test_mask | val_mask)),
            "val": np.flatnonzero(val_mask),
            "test": np.flatnonzero(test_mask),
        }
        _validate_split_sizes(result, n, test_size, val_size)
        return result

    if labels is None:
        idx = rng.permutation(n)
        n_test = int(n * test_size)
        test_idx = idx[:n_test]
        rem = idx[n_test:]
        n_val = int(rem.shape[0] * val_size)
        val_idx = rem[:n_val]
        train_idx = rem[n_val:]
        result = {"train": train_idx, "val": val_idx, "test": test_idx}
        _validate_split_sizes(result, n, test_size, val_size)
        return result

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
    result = {"train": train_idx, "val": val_idx, "test": test_idx}
    _validate_split_sizes(result, n, test_size, val_size)
    return result


def _validate_split_sizes(splits: Dict[str, np.ndarray], n: int, test_size: float, val_size: float) -> None:
    """Reject requested partitions that rounded down to empty arrays."""
    if len(splits["train"]) == 0 or (test_size > 0 and len(splits["test"]) == 0) or (val_size > 0 and len(splits["val"]) == 0):
        raise ValueError("Requested split produced an empty partition; increase n or adjust split sizes.")
    if sum(len(indices) for indices in splits.values()) != n:
        raise ValueError("Splits do not cover each input sample exactly once.")


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
    iq = np.asarray(iq)
    if iq.ndim != 1 or iq.shape[0] == 0 or not np.iscomplexobj(iq):
        raise ValueError("iq must be a non-empty one-dimensional complex array.")
    if not 0.0 <= aug_prob <= 1.0:
        raise ValueError("aug_prob must be between 0 and 1.")
    if rng is not None and seed is not None:
        raise ValueError("Pass either seed or rng, not both.")
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
        if iq.ndim != 2 or iq.shape[1] <= 0 or not np.iscomplexobj(iq) or device_id.ndim != 1 or len(iq) != len(device_id):
            raise ValueError("iq must be [N, T] and device_id must align with it.")
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

    def __init__(self, iq: np.ndarray, cfg: RFConfig, seed: Optional[int] = None):
        """Initialize two-view dataset.

        Args:
            iq: Complex waveforms with shape ``[N, T]``.
            cfg: Runtime config with augmentation settings.
        """
        if iq.ndim != 2 or iq.shape[1] <= 0 or not np.iscomplexobj(iq):
            raise ValueError("iq must be a non-empty complex array with shape [N, T].")
        self.iq = iq.astype(np.complex64)
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch component of deterministic augmentation randomness."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

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
        # Derive randomness from seed and index so repeated reads are stable,
        # including when a DataLoader changes worker assignment.
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        child_seeds = seed_sequence.spawn(2)
        aug_kwargs = dict(
            noise_std=self.cfg.noise_std,
            phase_jitter_rad=self.cfg.phase_jitter_rad,
            time_shift=self.cfg.time_shift,
            cfo_jitter_rad=self.cfg.cfo_jitter_rad,
            amplitude_jitter=self.cfg.amplitude_jitter,
            aug_prob=self.cfg.aug_prob,
        )
        v1 = augment_iq(base, rng=np.random.default_rng(child_seeds[0]), **aug_kwargs)
        v2 = augment_iq(base, rng=np.random.default_rng(child_seeds[1]), **aug_kwargs)
        return torch.from_numpy(v1), torch.from_numpy(v2)


def seeded_dataloader(dataset: Dataset, cfg: RFConfig, shuffle: bool = False) -> DataLoader:
    """Build a reproducibly shuffled DataLoader for an RF dataset.

    The helper is additive: existing callers can keep constructing DataLoaders
    directly, while experiments that need repeatable ordering can use this.
    """
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = cfg.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker if cfg.num_workers else None,
        generator=generator,
    )


class TwoViewLabeledIQDataset(Dataset):
    """Two augmented views plus a device label, for SupCon pretraining."""

    def __init__(self, iq: np.ndarray, device_id: np.ndarray, cfg: RFConfig,
                 seed: Optional[int] = None):
        if len(iq) != len(device_id):
            raise ValueError("iq and device_id must have equal length")
        self.inner = TwoViewIQDataset(iq, cfg, seed=seed)
        self.device_id = device_id.astype(np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        v1, v2 = self.inner[index]
        y = torch.tensor(self.device_id[index], dtype=torch.long)
        return v1, v2, y
