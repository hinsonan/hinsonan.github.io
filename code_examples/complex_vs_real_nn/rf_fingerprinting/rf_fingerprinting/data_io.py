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

    device_amp = 1.0 + 0.08 * rng.standard_normal(cfg.n_devices)
    device_phase = 0.6 * rng.standard_normal(cfg.n_devices)
    device_freq = 0.03 * rng.standard_normal(cfg.n_devices)

    iq = np.zeros((n, cfg.seq_len), dtype=np.complex64)
    device_id = np.zeros(n, dtype=np.int64)
    session_id = np.zeros(n, dtype=np.int64)

    for i in range(n):
        d = int(rng.integers(0, cfg.n_devices))
        s = int(rng.integers(0, cfg.n_sessions))

        base_sym = rng.choice(np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64), size=cfg.seq_len)
        session_phase = 0.2 * (s - 0.5 * (cfg.n_sessions - 1))

        rot = np.exp(1j * (device_phase[d] + session_phase + device_freq[d] * t))
        amp = device_amp[d] * (1.0 + 0.02 * rng.standard_normal())
        noise = cfg.noise_std * (
            rng.standard_normal(cfg.seq_len) + 1j * rng.standard_normal(cfg.seq_len)
        )

        x = amp * base_sym * rot + noise
        power = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-8)
        iq[i] = (x / power).astype(np.complex64)
        device_id[i] = d
        session_id[i] = s

    return {"iq": iq, "device_id": device_id, "session_id": session_id}


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
