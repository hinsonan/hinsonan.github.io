"""Dataset and signal-generation helpers for the AMC experiment."""

from typing import Dict, List

import numpy as np
from torchsig.signals.builders.constellation import constellation_modulator_baseband
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.transforms.functional import awgn as torchsig_awgn
from torchsig.transforms.functional import phase_offset as torchsig_phase_offset

try:
    from .config import ModClassConfig
except ImportError:
    from config import ModClassConfig


def _normalize_constellation(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.complex64)
    return (pts / np.sqrt(np.mean(np.abs(pts) ** 2))).astype(np.complex64)


CONSTELLATIONS = {
    name: _normalize_constellation(pts) for name, pts in all_symbol_maps.items()
}


def constellation(name: str) -> np.ndarray:
    return CONSTELLATIONS[name]


def generate_clean_burst(
    mod_name: str, config: ModClassConfig, rng: np.random.Generator
) -> np.ndarray:
    return constellation_modulator_baseband(
        constellation_name=mod_name,
        pulse_shape_name="rectangular",
        max_num_samples=config.burst_len,
        oversampling_rate_nominal=1,
        rng=rng,
    ).astype(np.complex64)


def rotate_burst(signal: np.ndarray, theta: float) -> np.ndarray:
    return torchsig_phase_offset(signal.astype(np.complex64), theta).astype(np.complex64)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.abs(signal) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_power_db = 10.0 * np.log10(noise_power)
    return torchsig_awgn(signal, noise_power_db=noise_power_db, rng=rng).astype(
        np.complex64
    )


def generate_burst(
    mod_name: str,
    config: ModClassConfig,
    rng: np.random.Generator,
    phase_low_deg: float,
    phase_high_deg: float,
    snr_db: float,
) -> "tuple[np.ndarray, np.float32]":
    symbols = generate_clean_burst(mod_name, config, rng)
    theta = rng.uniform(np.deg2rad(phase_low_deg), np.deg2rad(phase_high_deg))
    rotated = rotate_burst(symbols, theta)
    noisy = add_awgn(rotated, snr_db, rng)
    return noisy, np.float32(theta)


def generate_dataset(
    n: int,
    config: ModClassConfig,
    phase_low_deg: float,
    phase_high_deg: float,
    snr_db: float = None,
    seed_offset: int = 0,
) -> Dict:
    if snr_db is None:
        snr_db = config.snr_db
    rng = np.random.default_rng(config.seed + seed_offset)

    mods: List[str] = list(config.modulations)
    n_mod = len(mods)
    counts = [n // n_mod + (1 if i < n % n_mod else 0) for i in range(n_mod)]

    iq = np.empty((n, config.burst_len), dtype=np.complex64)
    label = np.empty(n, dtype=np.int64)
    theta = np.empty(n, dtype=np.float32)

    row = 0
    for mi, mod in enumerate(mods):
        for _ in range(counts[mi]):
            iq[row], theta[row] = generate_burst(
                mod, config, rng, phase_low_deg, phase_high_deg, snr_db
            )
            label[row] = mi
            row += 1

    perm = rng.permutation(n)
    return {
        "iq": iq[perm],
        "label": label[perm],
        "theta": theta[perm],
        "mods": mods,
        "snr_db": snr_db,
    }
