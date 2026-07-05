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
    noise_std: float,
    phase_jitter_rad: float,
    time_shift: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create one augmented view of an IQ waveform.

    Args:
        iq: Complex waveform of shape ``[T]``.
        noise_std: Noise standard deviation.
        phase_jitter_rad: Phase jitter range.
        time_shift: Max circular shift in samples.
        seed: Optional seed if ``rng`` is not provided.
        rng: Optional pre-built random generator.

    Returns:
        Augmented waveform.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    out = iq.astype(np.complex64, copy=True)
    out = apply_phase_jitter(out, phase_jitter_rad, rng)
    out = apply_time_shift(out, time_shift, rng)
    out = apply_awgn(out, noise_std, rng)
    return normalize_power(out)
