"""TorchSig QPSK generation with fixed per-emitter hardware fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy

import numpy as np
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.transforms import CarrierPhaseNoise, NonlinearAmplifier
from torchsig.utils.defaults import TorchSigDefaults


@dataclass(frozen=True)
class Emitter:
    """Fixed hardware characteristics of one synthetic emitter."""

    iq_gain_db: float
    iq_phase_rad: float
    phase_noise_deg: float
    pa_backoff: float
    pa_phase: float


def make_emitters(count: int, seed: int) -> list[Emitter]:
    """Draw one fixed hardware fingerprint per emitter."""
    rng = np.random.default_rng(seed)
    def spread(low: float, high: float) -> np.ndarray:
        return rng.permutation(np.linspace(low, high, count))

    gains = spread(-4.0, 4.0)
    phases = spread(-0.25, 0.25)
    phase_noise = spread(0.3, 5.0)
    backoff = spread(2.0, 10.0)
    pa_phase = spread(-0.3, 0.3)
    return [
        Emitter(
            iq_gain_db=float(gains[index]),
            iq_phase_rad=float(phases[index]),
            phase_noise_deg=float(phase_noise[index]),
            pa_backoff=float(backoff[index]),
            pa_phase=float(pa_phase[index]),
        )
        for index in range(count)
    ]


def emitter_table(emitters: list[Emitter]) -> list[dict[str, float | int]]:
    """Return emitter parameters in a notebook-friendly format."""
    return [{"emitter": index, **asdict(emitter)} for index, emitter in enumerate(emitters)]


def _generator(length: int, seed: int) -> TorchSigIterableDataset:
    metadata = TorchSigDefaults().default_dataset_metadata.copy()
    metadata.update(
        num_iq_samples_dataset=length,
        num_signals_min=1,
        num_signals_max=1,
        fft_size=64,
        fft_stride=64,
        sample_rate=1_000_000,
        noise_power_db=-100,
        snr_db_min=80,
        snr_db_max=80,
        signal_duration_in_samples_min=3 * length // 4,
        signal_duration_in_samples_max=3 * length // 4,
        bandwidth_min=250_000,
        bandwidth_max=250_000,
        signal_center_freq_min=0,
        signal_center_freq_max=0,
        frequency_min=-500_000,
        frequency_max=500_000,
    )
    # TorchSig 2.1.1 requires list form for one signal generator.
    return TorchSigIterableDataset(signal_generators=["qpsk"], seed=seed, **metadata)


def apply_iq_imbalance(iq: np.ndarray, gain_db: float, phase_rad: float) -> np.ndarray:
    """Apply differential I/Q gain and quadrature phase imbalance.

    TorchSig 2.1.1 applies its amplitude setting equally to I and Q. Reciprocal
    gains make this a true imbalance while avoiding an irrelevant common gain.
    The resulting real-linear map is equivalently alpha*x + beta*conj(x).
    """
    gain_ratio = 10 ** (gain_db / 20.0)
    i_gain = np.sqrt(gain_ratio)
    q_gain = 1 / i_gain
    i = i_gain * iq.real * np.exp(-0.5j * phase_rad)
    q = q_gain * iq.imag * np.exp(1j * (np.pi / 2 + 0.5 * phase_rad))
    return (i + q).astype(np.complex64)


def _apply_fingerprint(signal, emitter: Emitter, seed: int) -> np.ndarray:
    signal.data = apply_iq_imbalance(signal.data, emitter.iq_gain_db, emitter.iq_phase_rad)
    transforms = (
        CarrierPhaseNoise(phase_noise_degrees=(emitter.phase_noise_deg,) * 2, seed=seed + 1),
        NonlinearAmplifier(
            gain_range=(1.0, 1.0),
            psat_backoff_range=(emitter.pa_backoff,) * 2,
            phi_max_range=(emitter.pa_phase,) * 2,
            phi_slope_range=(emitter.pa_phase / 2,) * 2,
            seed=seed + 2,
        ),
    )
    for transform in transforms:
        signal = transform(signal)
    return signal.data.astype(np.complex64)


def add_channel(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add per-capture phase, gain, and AWGN nuisance variation."""
    iq = iq * rng.uniform(0.8, 1.2) * np.exp(1j * rng.uniform(-0.05, 0.05))
    signal_power = np.mean(np.abs(iq) ** 2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    output = iq + noise
    return (output / np.sqrt(np.mean(np.abs(output) ** 2) + 1e-12)).astype(np.complex64)


def make_dataset(
    emitters: list[Emitter],
    samples_per_emitter: int,
    length: int,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced emitter-identification dataset."""
    rng = np.random.default_rng(seed)
    generator = _generator(length, seed)
    samples, labels = [], []
    # Every emitter sees the same payload distribution, but train and test use
    # different payloads. This isolates hardware rather than content.
    for sample_id in range(samples_per_emitter):
        signal = next(generator)
        for emitter_id, emitter in enumerate(emitters):
            iq = _apply_fingerprint(deepcopy(signal), emitter, seed + 1000 * emitter_id + sample_id)
            samples.append(add_channel(iq, snr_db, rng))
            labels.append(emitter_id)
    order = rng.permutation(len(labels))
    return np.stack(samples)[order], np.asarray(labels, dtype=np.int64)[order]
