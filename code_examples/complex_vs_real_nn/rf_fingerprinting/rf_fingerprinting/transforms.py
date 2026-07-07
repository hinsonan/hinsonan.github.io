"""Signal transforms and augmentations for RF IQ captures."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_phase_jitter(iq: np.ndarray, max_jitter_rad: float, rng: np.random.Generator) -> np.ndarray:
    """Apply random global phase jitter.

    Args:
        iq: Complex waveform of shape ``[T]``.
        max_jitter_rad: Max absolute phase offset in radians.
        rng: NumPy random generator.

    Returns:
        Phase-jittered waveform.
    """
    phi = rng.uniform(-max_jitter_rad, max_jitter_rad)
    return iq * np.exp(1j * phi).astype(np.complex64)


def apply_awgn(iq: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex white Gaussian noise.

    Args:
        iq: Complex waveform.
        noise_std: Standard deviation per real/imag component.
        rng: NumPy random generator.

    Returns:
        Noisy waveform.
    """
    noise = noise_std * (rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def apply_time_shift(iq: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Apply random circular time shift.

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


def apply_cfo(iq: np.ndarray, max_cfo_per_sample: float, rng: np.random.Generator) -> np.ndarray:
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


def apply_amplitude_jitter(iq: np.ndarray, max_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a random global amplitude scale.

    Args:
        iq: Complex waveform.
        max_scale: Max relative amplitude perturbation (e.g., 0.1 = ±10%).
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
    phase_jitter_rad: float = 0.35,
    time_shift: int = 8,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    cfo_jitter_rad: float = 0.0,
    amplitude_jitter: float = 0.0,
    aug_prob: float = 1.0,
) -> np.ndarray:
    """Create one augmented view of an IQ waveform.

    Each transform is applied independently with probability ``aug_prob``
    so that the model sees a variety of augmentation combinations.

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
