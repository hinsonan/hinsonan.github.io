"""Data loading and reproducible synthetic RF fingerprint generation.

The synthetic path is deliberately layered: a burst waveform is passed through
persistent device hardware, a session channel, and a receiver front end.  The
returned metadata makes it possible to audit whether an evaluation split is
testing device identity or merely memorizing a nuisance condition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from .config import RFConfig
except ImportError:
    from config import RFConfig


_METADATA_DTYPES = {
    "session_id": np.int64,
    "receiver_id": np.int64,
    "channel_id": np.int64,
    "waveform_id": np.int64,
    "snr_db": np.float32,
}


def _categorical_id(values: np.ndarray, name: str, n: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError(f"'{name}' must have shape [N] and align with 'iq'.")
    if raw.dtype.kind == "f" and (not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw))):
        raise ValueError(f"'{name}' contains invalid non-integral IDs.")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"'{name}' must contain categorical integer IDs.")
    try:
        return raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{name}' must contain categorical integer IDs.") from exc


def _validate_npz_dict(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Validate and retain optional per-sample metadata in an NPZ dataset."""
    if "iq" not in data or "device_id" not in data:
        raise ValueError("NPZ file must contain keys: 'iq' and 'device_id'.")

    raw_iq = np.asarray(data["iq"])
    if not np.iscomplexobj(raw_iq):
        raise ValueError("'iq' must contain complex-valued samples.")
    iq = raw_iq.astype(np.complex64)
    device_id = _categorical_id(data["device_id"], "device_id", np.asarray(data["iq"]).shape[0])
    if iq.ndim != 2 or iq.shape[1] <= 0:
        raise ValueError("'iq' must have shape [N, T] with T > 0.")
    if device_id.ndim != 1 or device_id.shape[0] != iq.shape[0]:
        raise ValueError("'device_id' must have shape [N] and align with 'iq'.")

    result: Dict[str, np.ndarray] = {"iq": iq, "device_id": device_id}
    unavailable = []
    for name, dtype in _METADATA_DTYPES.items():
        values = data.get(name)
        if values is None:
            # Keep the legacy array-shaped API, but make unavailable metadata
            # impossible to mistake for a valid group.
            values = np.full(iq.shape[0], -1 if np.issubdtype(dtype, np.integer) else np.nan, dtype=dtype)
            unavailable.append(name)
        elif name != "snr_db":
            values = _categorical_id(values, name, iq.shape[0])
        else:
            values = np.asarray(values, dtype=dtype)
        if values.ndim != 1 or values.shape[0] != iq.shape[0]:
            raise ValueError(f"'{name}' must have shape [N] and align with 'iq'.")
        result[name] = values
    result["metadata_unavailable"] = np.asarray(unavailable, dtype="U32")

    # Keep other simple, aligned metadata columns available to evaluators.
    for name, values in data.items():
        if name in result or name in {"iq", "device_id"}:
            continue
        values = np.asarray(values)
        if values.ndim == 1 and values.shape[0] == iq.shape[0] and values.dtype != object:
            result[name] = values
    return result


def _qam_alphabet(waveform_id: int) -> np.ndarray:
    """Return a normalized BPSK, QPSK, or 16-QAM alphabet."""
    if waveform_id == 0:
        return np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
    if waveform_id == 1:
        return np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2)
    points = np.array([-3, -1, 1, 3], dtype=np.float32)
    return np.array([i + 1j * q for i in points for q in points], dtype=np.complex64) / np.sqrt(10)


def _make_burst(seq_len: int, waveform_id: int, rng: np.random.Generator) -> np.ndarray:
    """Create a short shaped burst with a deterministic pilot/preamble."""
    symbols = rng.choice(_qam_alphabet(waveform_id), size=seq_len).astype(np.complex64)
    # A repeated pilot makes the waveform less like independent complex noise.
    pilot_len = min(16, seq_len)
    pilot = np.exp(1j * np.pi * np.arange(pilot_len, dtype=np.float32) / 2).astype(np.complex64)
    symbols[:pilot_len] = pilot
    taps = np.array([0.08, 0.24, 0.36, 0.24, 0.08], dtype=np.float32)
    return _same_length_convolve(symbols, taps)


def _same_length_convolve(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve while always returning the input signal length."""
    full = np.convolve(signal, taps, mode="full")
    start = (len(taps) - 1) // 2
    return full[start:start + signal.shape[0]].astype(np.complex64)


def _apply_device_effects(x: np.ndarray, params: Dict[str, np.ndarray], device: int,
                          t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply persistent oscillator, IQ, PA, and memory impairments."""
    i = x.real * params["i_gain"][device]
    q = x.imag * params["q_gain"][device]
    q = q * np.cos(params["iq_skew"][device]) + i * np.sin(params["iq_skew"][device])
    x = (i + 1j * q).astype(np.complex64)
    delayed = np.concatenate((np.zeros(1, dtype=x.dtype), x[:-1]))
    x = x + params["memory"][device] * delayed
    amplitude = np.abs(x)
    x = x * (1.0 + params["pa"][device] * amplitude**2) / (1.0 + 0.12 * amplitude**2)
    phase = params["cfo"][device] * t + np.cumsum(
        rng.normal(0.0, params["phase_noise"][device], size=t.shape[0])
    )
    return (x * np.exp(1j * phase) + params["dc"][device]).astype(np.complex64)


def _apply_channel(x: np.ndarray, channel: Dict[str, np.ndarray], session: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Apply multipath fading, slow phase drift, and session-specific Doppler."""
    x = _same_length_convolve(x, channel["taps"][session])
    t = np.arange(x.shape[0], dtype=np.float32)
    phase = channel["phase"][session] + channel["doppler"][session] * t
    x = x * np.exp(1j * phase).astype(np.complex64)
    return x * np.complex64(1.0 + rng.normal(0.0, 0.025))


def _apply_receiver(x: np.ndarray, receiver: Dict[str, np.ndarray], receiver_id: int,
                    rng: np.random.Generator, quantization_scale: float = 2.0,
                    quantization_bits: int = 10) -> tuple[np.ndarray, float]:
    """Apply receiver gain/imbalance, AGC, quantization, and thermal noise."""
    i = x.real * receiver["i_gain"][receiver_id]
    q = x.imag * receiver["q_gain"][receiver_id]
    x = (i + 1j * q + receiver["dc"][receiver_id]).astype(np.complex64)
    x = x * np.exp(1j * receiver["phase"][receiver_id]).astype(np.complex64)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-8) * receiver["agc"][receiver_id]
    snr_db = float(receiver["snr_db"][receiver_id] + rng.normal(0.0, 1.0))
    noise_std = np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))
    noise = noise_std * (rng.standard_normal(x.shape[0]) + 1j * rng.standard_normal(x.shape[0]))
    x = x + noise
    actual_snr_db = 10.0 * np.log10(
        (np.mean(np.abs(x - noise) ** 2) + 1e-12) /
        (np.mean(np.abs(noise) ** 2) + 1e-12)
    )
    # A fixed full-scale range makes clipping and quantization comparable across samples.
    levels = 2**quantization_bits - 1
    real = np.clip(np.round((x.real / quantization_scale + 1.0) * levels / 2.0), 0, levels)
    imag = np.clip(np.round((x.imag / quantization_scale + 1.0) * levels / 2.0), 0, levels)
    real = real * 2.0 / levels - 1.0
    imag = imag * 2.0 / levels - 1.0
    return (quantization_scale * (real + 1j * imag)).astype(np.complex64), float(actual_snr_db)


def generate_synthetic_rf_data(cfg: RFConfig) -> Dict[str, np.ndarray]:
    """Generate realistic, deterministic synthetic RF fingerprints.

    Device parameters are sampled once per device and therefore remain stable
    across sessions and receiver conditions.  Session and receiver parameters
    are independent nuisance factors and are returned as metadata.
    """
    rng = np.random.default_rng(cfg.seed)
    n, length = cfg.n_samples, cfg.seq_len
    t = np.arange(length, dtype=np.float32)
    n_receivers = cfg.n_receivers

    device = {
        "i_gain": 1.0 + 0.06 * rng.standard_normal(cfg.n_devices),
        "q_gain": 1.0 + 0.06 * rng.standard_normal(cfg.n_devices),
        "iq_skew": 0.06 * rng.standard_normal(cfg.n_devices),
        "cfo": rng.uniform(-0.012, 0.012, cfg.n_devices),
        "phase_noise": rng.uniform(0.0008, 0.0025, cfg.n_devices),
        "pa": rng.uniform(0.03, 0.11, cfg.n_devices),
        "memory": rng.uniform(-0.06, 0.06, cfg.n_devices),
        "dc": (0.025 * (rng.standard_normal(cfg.n_devices) + 1j * rng.standard_normal(cfg.n_devices))).astype(np.complex64),
    }
    session_channel = {
        "taps": (0.10 * (rng.standard_normal((cfg.n_sessions, 5)) + 1j * rng.standard_normal((cfg.n_sessions, 5)))).astype(np.complex64),
        "phase": rng.uniform(-np.pi, np.pi, cfg.n_sessions),
        "doppler": rng.uniform(-0.001, 0.001, cfg.n_sessions),
    }
    session_channel["taps"][:, 2] += 1.0
    channel = {
        "taps": (0.12 * (rng.standard_normal((cfg.n_channels, 5)) + 1j * rng.standard_normal((cfg.n_channels, 5)))).astype(np.complex64),
        "phase": rng.uniform(-np.pi, np.pi, cfg.n_channels),
        "doppler": rng.uniform(-0.002, 0.002, cfg.n_channels),
    }
    channel["taps"][:, 2] += 1.0
    receiver = {
        "i_gain": 1.0 + 0.025 * rng.standard_normal(n_receivers),
        "q_gain": 1.0 + 0.025 * rng.standard_normal(n_receivers),
        "phase": rng.uniform(-0.08, 0.08, n_receivers),
        "dc": (0.012 * (rng.standard_normal(n_receivers) + 1j * rng.standard_normal(n_receivers))).astype(np.complex64),
        "agc": rng.uniform(0.88, 1.12, n_receivers),
        "snr_db": rng.uniform(16.0, 27.0, n_receivers),
    }

    counts = [n // cfg.n_devices + (i < n % cfg.n_devices) for i in range(cfg.n_devices)]
    output = {
        "iq": np.zeros((n, length), dtype=np.complex64),
        "device_id": np.zeros(n, dtype=np.int64),
        "session_id": np.zeros(n, dtype=np.int64),
        "receiver_id": np.zeros(n, dtype=np.int64),
        "channel_id": np.zeros(n, dtype=np.int64),
        "waveform_id": np.zeros(n, dtype=np.int64),
        "snr_db": np.zeros(n, dtype=np.float32),
    }
    row = 0
    for device_id, count in enumerate(counts):
        for _ in range(count):
            session_id = int(rng.integers(cfg.n_sessions))
            channel_id = int(rng.integers(cfg.n_channels))
            receiver_id = int(rng.integers(n_receivers))
            waveform_id = int(rng.integers(3))
            x = _make_burst(length, waveform_id, rng)
            x = _apply_device_effects(x, device, device_id, t, rng)
            x = _apply_channel(x, session_channel, session_id, rng)
            x = _apply_channel(x, channel, channel_id, rng)
            x, snr_db = _apply_receiver(
                x, receiver, receiver_id, rng, cfg.quantization_scale, cfg.quantization_bits
            )
            if cfg.normalize_output:
                x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-8)
            output["iq"][row] = x
            output["device_id"][row] = device_id
            output["session_id"][row] = session_id
            output["receiver_id"][row] = receiver_id
            output["channel_id"][row] = channel_id
            output["waveform_id"][row] = waveform_id
            output["snr_db"][row] = snr_db
            row += 1

    perm = rng.permutation(n)
    return {name: values[perm] for name, values in output.items()}


def load_or_generate_npz(cfg: RFConfig, dataset_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load an NPZ dataset or use the synthetic RF fallback."""
    path = dataset_path if dataset_path is not None else cfg.dataset_path
    if path:
        p = Path(path)
        if p.exists():
            with np.load(p) as npz:
                return _validate_npz_dict({k: npz[k] for k in npz.files})
        print(f"Warning: dataset_path '{p}' does not exist. Using synthetic fallback data.")
    return generate_synthetic_rf_data(cfg)
