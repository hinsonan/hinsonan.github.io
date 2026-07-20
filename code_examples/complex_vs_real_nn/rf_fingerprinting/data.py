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
                          t: np.ndarray, rng: np.random.Generator,
                          impairments=None) -> np.ndarray:
    """Apply persistent oscillator, IQ, PA, and memory impairments."""
    if impairments is None or impairments.iq_imbalance:
        i = x.real * params["i_gain"][device]
        q = x.imag * params["q_gain"][device]
        q = q * np.cos(params["iq_skew"][device]) + i * np.sin(params["iq_skew"][device])
    else:
        i, q = x.real, x.imag
    x = (i + 1j * q).astype(np.complex64)
    if impairments is None or impairments.pa_memory:
        delayed = np.concatenate((np.zeros(1, dtype=x.dtype), x[:-1]))
        x = x + params["memory"][device] * delayed
    if impairments is None or impairments.pa_nonlinearity:
        amplitude = np.abs(x)
        x = x * (1.0 + params["pa"][device] * amplitude**2) / (1.0 + 0.12 * amplitude**2)
    phase = np.zeros_like(t)
    if impairments is None or impairments.cfo:
        phase += params["cfo"][device] * t
    if impairments is None or impairments.phase_noise:
        phase += np.cumsum(rng.normal(0.0, params["phase_noise"][device], size=t.shape[0]))
    if impairments is None or impairments.dc_offset:
        x = x + params["dc"][device]
    return (x * np.exp(1j * phase)).astype(np.complex64)


def _apply_channel(x: np.ndarray, channel: Dict[str, np.ndarray], session: int,
                    rng: np.random.Generator, enabled: bool = True) -> np.ndarray:
    """Apply multipath fading, slow phase drift, and session-specific Doppler."""
    if not enabled:
        return x
    x = _same_length_convolve(x, channel["taps"][session])
    t = np.arange(x.shape[0], dtype=np.float32)
    phase = channel["phase"][session] + channel["doppler"][session] * t
    x = x * np.exp(1j * phase).astype(np.complex64)
    return x * np.complex64(1.0 + rng.normal(0.0, 0.025))


def _uniform_std(bounds) -> float:
    """Standard deviation of a uniform range, floored for degenerate ranges."""
    return max((float(bounds[1]) - float(bounds[0])) / float(np.sqrt(12.0)), 1e-12)


def _standardized_device_matrix(params: Dict[str, np.ndarray], impairments) -> np.ndarray:
    """Return per-device impairment vectors scaled by their sampling spread.

    Distances in this space are measured in impairment-standard-deviation
    units, which makes a single separation threshold meaningful across the
    different impairment profiles.
    """
    cols = [
        np.asarray(params["i_gain"], dtype=np.float64) / max(impairments.iq_gain_std, 1e-12),
        np.asarray(params["q_gain"], dtype=np.float64) / max(impairments.iq_gain_std, 1e-12),
        np.asarray(params["iq_skew"], dtype=np.float64) / max(impairments.iq_skew_std_rad, 1e-12),
        np.asarray(params["cfo"], dtype=np.float64) / _uniform_std(impairments.cfo_range_rad),
        np.asarray(params["phase_noise"], dtype=np.float64) / _uniform_std(impairments.phase_noise_range),
        np.asarray(params["pa"], dtype=np.float64) / _uniform_std(impairments.pa_range),
        np.asarray(params["memory"], dtype=np.float64) / _uniform_std(impairments.memory_range),
    ]
    if impairments.dc_offset:
        cols.append(np.asarray(params["dc"], dtype=np.complex128).real / max(impairments.dc_std, 1e-12))
        cols.append(np.asarray(params["dc"], dtype=np.complex128).imag / max(impairments.dc_std, 1e-12))
    return np.stack([np.atleast_1d(col) for col in cols], axis=1)


def _draw_single_device_params(rng: np.random.Generator, impairments) -> Dict[str, np.ndarray]:
    """Draw one device's persistent impairment parameters."""
    return {
        "i_gain": np.float64(1.0 + impairments.iq_gain_std * rng.standard_normal()),
        "q_gain": np.float64(1.0 + impairments.iq_gain_std * rng.standard_normal()),
        "iq_skew": np.float64(impairments.iq_skew_std_rad * rng.standard_normal()),
        "cfo": np.float64(rng.uniform(*impairments.cfo_range_rad)),
        "phase_noise": np.float64(rng.uniform(*impairments.phase_noise_range)),
        "pa": np.float64(rng.uniform(*impairments.pa_range)),
        "memory": np.float64(rng.uniform(*impairments.memory_range)),
        "dc": np.complex64(impairments.dc_std * (rng.standard_normal() + 1j * rng.standard_normal())),
    }


def _enforce_unknown_separation(device: Dict[str, np.ndarray], rng: np.random.Generator,
                                impairments, known_count: int, min_separation: float,
                                max_retries: int = 500) -> Dict[str, np.ndarray]:
    """Rejection-sample unknown devices away from known impairment space.

    Unknown devices (index >= ``known_count``) are redrawn until their
    standardized impairment vector is at least ``min_separation`` from every
    known device. This keeps the open-set benchmark well-posed: without the
    constraint, an unknown device can land arbitrarily close to a known one
    and become structurally impossible to reject.
    """
    n_devices = len(device["cfo"])
    known = {name: values[:known_count] for name, values in device.items()}
    known_mat = _standardized_device_matrix(known, impairments)
    for unknown_idx in range(known_count, n_devices):
        for _ in range(max_retries):
            candidate = _draw_single_device_params(rng, impairments)
            cand_vec = _standardized_device_matrix(candidate, impairments)[0]
            if np.linalg.norm(known_mat - cand_vec[None, :], axis=1).min() >= min_separation:
                for name, value in candidate.items():
                    device[name][unknown_idx] = value
                break
        else:
            raise ValueError(
                f"Could not sample unknown device index {unknown_idx} with min separation "
                f"{min_separation} after {max_retries} retries; lower "
                "unknown_min_separation or widen the device impairment ranges."
            )
    return device


def _nearest_known_distances(device: Dict[str, np.ndarray], impairments,
                             known_count: int) -> np.ndarray:
    """Return per-device distance to the nearest known device in std units.

    Known devices report the distance to the nearest *other* known device so
    the array is a direct indicator of how packed the enrolled devices are.
    """
    mat = _standardized_device_matrix(device, impairments)
    known = mat[:known_count]
    dists = np.zeros(mat.shape[0], dtype=np.float64)
    for idx in range(mat.shape[0]):
        reference = np.delete(known, idx, axis=0) if idx < known_count else known
        dists[idx] = np.linalg.norm(reference - mat[idx], axis=1).min() if len(reference) else 0.0
    return dists


def _apply_receiver(x: np.ndarray, receiver: Dict[str, np.ndarray], receiver_id: int,
                    rng: np.random.Generator, quantization_scale: float = 2.0,
                    quantization_bits: int = 10, impairments=None) -> tuple[np.ndarray, float]:
    """Apply receiver gain/imbalance, AGC, quantization, and thermal noise."""
    if impairments is None or impairments.receiver:
        i = x.real * receiver["i_gain"][receiver_id]
        q = x.imag * receiver["q_gain"][receiver_id]
        x = (i + 1j * q + receiver["dc"][receiver_id]).astype(np.complex64)
        x = x * np.exp(1j * receiver["phase"][receiver_id]).astype(np.complex64)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-8) * receiver["agc"][receiver_id]
    snr_db = float(receiver["snr_db"][receiver_id] + rng.normal(0.0, 1.0))
    if impairments is not None and not impairments.awgn:
        noise = np.zeros(x.shape[0], dtype=np.complex64)
    else:
        noise_std = np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))
        noise = noise_std * (rng.standard_normal(x.shape[0]) + 1j * rng.standard_normal(x.shape[0]))
        x = x + noise
    actual_snr_db = 10.0 * np.log10(
        (np.mean(np.abs(x - noise) ** 2) + 1e-12) /
        (np.mean(np.abs(noise) ** 2) + 1e-12)
    )
    # A fixed full-scale range makes clipping and quantization comparable across samples.
    if impairments is None or impairments.quantization:
        levels = 2**quantization_bits - 1
        real = np.clip(np.round((x.real / quantization_scale + 1.0) * levels / 2.0), 0, levels)
        imag = np.clip(np.round((x.imag / quantization_scale + 1.0) * levels / 2.0), 0, levels)
        real = real * 2.0 / levels - 1.0
        imag = imag * 2.0 / levels - 1.0
        x = quantization_scale * (real + 1j * imag)
    return x.astype(np.complex64), float(actual_snr_db)


def generate_synthetic_rf_data(cfg: RFConfig) -> Dict[str, np.ndarray]:
    """Generate realistic, deterministic synthetic RF fingerprints.

    Device parameters are sampled once per device and therefore remain stable
    across sessions and receiver conditions.  Session and receiver parameters
    are independent nuisance factors and are returned as metadata.

    When ``cfg.known_device_count`` is set, devices with index greater or
    equal to it are treated as unknown: they may be rejection-sampled to
    keep ``cfg.unknown_min_separation`` impairment-std distance from every
    known device, and a per-sample ``dist_to_nearest_known`` metadata column
    is populated for open-set reporting.
    """
    rng = np.random.default_rng(cfg.seed)
    device_impairments = cfg.device_impairments
    nuisance_impairments = cfg.nuisance_impairments
    n, length = cfg.n_samples, cfg.seq_len
    t = np.arange(length, dtype=np.float32)
    n_receivers = cfg.n_receivers

    device = {
        "i_gain": 1.0 + device_impairments.iq_gain_std * rng.standard_normal(cfg.n_devices),
        "q_gain": 1.0 + device_impairments.iq_gain_std * rng.standard_normal(cfg.n_devices),
        "iq_skew": device_impairments.iq_skew_std_rad * rng.standard_normal(cfg.n_devices),
        "cfo": rng.uniform(*device_impairments.cfo_range_rad, cfg.n_devices),
        "phase_noise": rng.uniform(*device_impairments.phase_noise_range, cfg.n_devices),
        "pa": rng.uniform(*device_impairments.pa_range, cfg.n_devices),
        "memory": rng.uniform(*device_impairments.memory_range, cfg.n_devices),
        "dc": (device_impairments.dc_std * (rng.standard_normal(cfg.n_devices) + 1j * rng.standard_normal(cfg.n_devices))).astype(np.complex64),
    }
    per_device_dist = None
    if cfg.known_device_count is not None:
        if not 1 <= cfg.known_device_count < cfg.n_devices:
            raise ValueError("known_device_count must be in [1, n_devices).")
        if cfg.unknown_min_separation > 0:
            device = _enforce_unknown_separation(
                device, rng, device_impairments, cfg.known_device_count,
                cfg.unknown_min_separation,
            )
        per_device_dist = _nearest_known_distances(device, device_impairments, cfg.known_device_count)
    session_channel = {
        "taps": (nuisance_impairments.session_tap_scale * (rng.standard_normal((cfg.n_sessions, 5)) + 1j * rng.standard_normal((cfg.n_sessions, 5)))).astype(np.complex64),
        "phase": rng.uniform(-np.pi, np.pi, cfg.n_sessions),
        "doppler": rng.uniform(-0.001, 0.001, cfg.n_sessions),
    }
    session_channel["taps"][:, 2] += 1.0
    channel = {
        "taps": (nuisance_impairments.channel_tap_scale * (rng.standard_normal((cfg.n_channels, 5)) + 1j * rng.standard_normal((cfg.n_channels, 5)))).astype(np.complex64),
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
        "snr_db": rng.uniform(*nuisance_impairments.snr_db_range, n_receivers),
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
        "dist_to_nearest_known": np.zeros(n, dtype=np.float32),
    }
    row = 0
    for device_id, count in enumerate(counts):
        for _ in range(count):
            session_id = int(rng.integers(cfg.n_sessions))
            channel_id = int(rng.integers(cfg.n_channels))
            receiver_id = int(rng.integers(n_receivers))
            waveform_id = int(rng.integers(3)) if nuisance_impairments.waveform_variation else 0
            x = _make_burst(length, waveform_id, rng)
            x = _apply_device_effects(x, device, device_id, t, rng, device_impairments)
            x = _apply_channel(x, session_channel, session_id, rng, nuisance_impairments.session_channel)
            x = _apply_channel(x, channel, channel_id, rng, nuisance_impairments.channel)
            x, snr_db = _apply_receiver(
                x, receiver, receiver_id, rng, cfg.quantization_scale, cfg.quantization_bits,
                nuisance_impairments
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
            output["dist_to_nearest_known"][row] = (
                per_device_dist[device_id] if per_device_dist is not None else np.nan
            )
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
