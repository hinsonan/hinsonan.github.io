"""Data loading and synthetic fallback generation for RF fingerprinting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import RFConfig


def _validate_npz_dict(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Validate required keys and dtypes for NPZ-loaded data.

    Args:
        data: Raw arrays dictionary.

    Returns:
        Normalized dictionary with required keys.

    Raises:
        ValueError: If required arrays are missing or malformed.
    """
    if "iq" not in data or "device_id" not in data:
        raise ValueError("NPZ file must contain keys: 'iq' and 'device_id'.")

    iq = np.asarray(data["iq"], dtype=np.complex64)
    device_id = np.asarray(data["device_id"], dtype=np.int64)
    session_id = data.get("session_id")
    if session_id is None:
        session_id = np.zeros(device_id.shape[0], dtype=np.int64)
    else:
        session_id = np.asarray(session_id, dtype=np.int64)

    if iq.ndim != 2:
        raise ValueError("'iq' must have shape [N, T].")
    if device_id.ndim != 1 or device_id.shape[0] != iq.shape[0]:
        raise ValueError("'device_id' must have shape [N] and align with 'iq'.")
    if session_id.ndim != 1 or session_id.shape[0] != iq.shape[0]:
        raise ValueError("'session_id' must have shape [N] and align with 'iq'.")

    return {"iq": iq, "device_id": device_id, "session_id": session_id}


def generate_synthetic_rf_data(cfg: RFConfig) -> Dict[str, np.ndarray]:
    """Generate lightweight synthetic RF fingerprints.

    Each device gets a small signature in amplitude/phase/frequency response,
    plus per-session perturbations.

    Args:
        cfg: Runtime config.

    Returns:
        Dictionary with keys ``iq``, ``device_id``, ``session_id``.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples
    t = np.arange(cfg.seq_len, dtype=np.float32)

    constellations = [
        np.array([-1 + 0j, 1 + 0j], dtype=np.complex64),
        np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2),
        np.array(
            [
                -3 - 3j,
                -3 - 1j,
                -3 + 1j,
                -3 + 3j,
                -1 - 3j,
                -1 - 1j,
                -1 + 1j,
                -1 + 3j,
                1 - 3j,
                1 - 1j,
                1 + 1j,
                1 + 3j,
                3 - 3j,
                3 - 1j,
                3 + 1j,
                3 + 3j,
            ],
            dtype=np.complex64,
        )
        / np.sqrt(10),
    ]

    # Device-specific hardware fingerprints.
    iq_gain = 1.0 + 0.08 * rng.standard_normal((cfg.n_devices, 2))
    iq_phase_skew = 0.08 * rng.standard_normal(cfg.n_devices)
    cfo = 0.01 * rng.standard_normal(cfg.n_devices)
    nonlinearity = 0.03 + 0.05 * rng.random(cfg.n_devices)
    dc_offset = 0.04 * (
        rng.standard_normal(cfg.n_devices) + 1j * rng.standard_normal(cfg.n_devices)
    )
    pa_memory = (
        0.18
        * (rng.standard_normal((cfg.n_devices, 5)) + 1j * rng.standard_normal((cfg.n_devices, 5)))
    ).astype(np.complex64)
    pa_memory[:, 2] += 1.0

    # Session-level propagation/channel perturbations.
    session_phase = rng.uniform(-np.pi, np.pi, size=cfg.n_sessions)
    session_snr_db = rng.uniform(14.0, 24.0, size=cfg.n_sessions)
    session_channel = (
        0.2 * (rng.standard_normal((cfg.n_sessions, 3)) + 1j * rng.standard_normal((cfg.n_sessions, 3)))
    ).astype(np.complex64)
    session_channel[:, 1] += 1.0

    counts = [n // cfg.n_devices + (1 if i < n % cfg.n_devices else 0) for i in range(cfg.n_devices)]
    iq = np.zeros((n, cfg.seq_len), dtype=np.complex64)
    device_id = np.zeros(n, dtype=np.int64)
    session_id = np.zeros(n, dtype=np.int64)

    row = 0
    for d in range(cfg.n_devices):
        for _ in range(counts[d]):
            s = int(rng.integers(0, cfg.n_sessions))
            alphabet = constellations[int(rng.integers(0, len(constellations)))]
            base = rng.choice(alphabet, size=cfg.seq_len).astype(np.complex64)

            # Device PA memory effect.
            x = np.convolve(base, pa_memory[d], mode="same").astype(np.complex64)

            # Device IQ imbalance and phase skew.
            i_part = x.real * iq_gain[d, 0]
            q_part = x.imag * iq_gain[d, 1]
            skew = iq_phase_skew[d]
            q_mix = q_part * np.cos(skew) + i_part * np.sin(skew)
            x = (i_part + 1j * q_mix).astype(np.complex64)

            # Device nonlinearity + CFO + DC offset.
            x = x + nonlinearity[d] * (np.abs(x) ** 2) * x
            init_phase = rng.uniform(-np.pi, np.pi)
            x = x * np.exp(1j * (init_phase + cfo[d] * t)).astype(np.complex64)
            x = x + dc_offset[d]

            # Session channel and nuisance effects.
            x = np.convolve(x, session_channel[s], mode="same").astype(np.complex64)
            x = x * np.exp(1j * session_phase[s]).astype(np.complex64)

            snr_db = float(session_snr_db[s] + rng.normal(0.0, 1.0))
            signal_power = float(np.mean(np.abs(x) ** 2) + 1e-8)
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise_std = np.sqrt(noise_power / 2.0)
            noise = noise_std * (
                rng.standard_normal(cfg.seq_len) + 1j * rng.standard_normal(cfg.seq_len)
            )
            x = x + noise.astype(np.complex64)

            power = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-8)
            iq[row] = (x / power).astype(np.complex64)
            device_id[row] = d
            session_id[row] = s
            row += 1

    perm = rng.permutation(n)
    return {
        "iq": iq[perm],
        "device_id": device_id[perm],
        "session_id": session_id[perm],
    }


def load_or_generate_npz(cfg: RFConfig, dataset_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load RF data from NPZ or generate synthetic fallback.

    Args:
        cfg: Runtime config.
        dataset_path: Optional override path to NPZ file.

    Returns:
        Dictionary with keys ``iq``, ``device_id``, ``session_id``.
    """
    path = dataset_path if dataset_path is not None else cfg.dataset_path
    if path:
        p = Path(path)
        if p.exists():
            npz = np.load(p)
            return _validate_npz_dict({k: npz[k] for k in npz.files})
        print(f"Warning: dataset_path '{p}' does not exist. Using synthetic fallback data.")
    return generate_synthetic_rf_data(cfg)
