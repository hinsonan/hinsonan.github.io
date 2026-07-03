"""Core reusable code for the rotation-generalization AMC experiment."""
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsig.signals.builders.constellation import constellation_modulator_baseband
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.transforms.functional import awgn as torchsig_awgn
from torchsig.transforms.functional import phase_offset as torchsig_phase_offset


@dataclass
class ModClassConfig:
    """Configuration for rotation-invariant modulation classification."""

    burst_len: int = 128
    modulations: tuple = ("bpsk", "qpsk", "8psk", "16qam")

    train_phase_deg: float = 15.0
    full_phase_deg: float = 180.0

    snr_db: float = 10.0

    n_train: int = 12000
    n_val: int = 4000
    seed: int = 7

    complex_channels: tuple = (24, 48, 48)
    real_channels: tuple = (32, 64, 64)
    kernel_size: int = 7
    stride: int = 2
    hidden_dim: int = 128
    moment_hidden_dim: int = 48
    moment_orders: tuple = (2, 4, 8)

    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    out_dir: str = "trained_modclass"

    @property
    def n_classes(self) -> int:
        return len(self.modulations)


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


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size
        std = 1.0 / math.sqrt(2.0 * fan_in)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, dtype=torch.complex64)
            * std
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding
        )


class modReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        return torch.relu(mag + self.b) * z / (mag + 1e-8)


class ComplexModClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (24, 48, 48),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.features = nn.ModuleList()
        prev = 1
        for ch in channels:
            self.features.append(
                nn.Sequential(
                    ComplexConv1d(
                        prev, ch, kernel_size, stride=stride, padding=pad, bias=False
                    ),
                    modReLU(),
                )
            )
            prev = ch

        self.head = nn.Sequential(
            nn.Linear(2 * channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    @staticmethod
    def _magnitude_pool(h: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(h)
        mean = mag.mean(dim=2)
        std = mag.std(dim=2)
        return torch.cat([mean, std], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        for block in self.features:
            h = block(h)
        pooled = self._magnitude_pool(h)
        return self.head(pooled)


class ComplexMomentClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (24, 48, 48),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
        moment_orders: tuple = (2, 4, 8),
    ):
        super().__init__()
        pad = kernel_size // 2
        self.moment_orders = moment_orders

        self.features = nn.ModuleList()
        prev = 1
        for ch in channels:
            self.features.append(
                nn.Sequential(
                    ComplexConv1d(
                        prev, ch, kernel_size, stride=stride, padding=pad, bias=False
                    ),
                    modReLU(),
                )
            )
            prev = ch

        n_stats = 2 + len(moment_orders)
        self.head = nn.Sequential(
            nn.Linear(n_stats * channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def _moment_pool(self, h: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(h)
        stats = [mag.mean(dim=2), mag.std(dim=2)]

        unit = h / (mag + 1e-8)
        for order in self.moment_orders:
            stats.append(torch.abs(torch.mean(unit ** order, dim=2)))
        return torch.cat(stats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        for block in self.features:
            h = block(h)
        pooled = self._moment_pool(h)
        return self.head(pooled)


class RealModClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (32, 64, 64),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.features = nn.ModuleList()
        prev = 2
        for ch in channels:
            self.features.append(
                nn.Sequential(
                    nn.Conv1d(prev, ch, kernel_size, stride=stride, padding=pad),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                )
            )
            prev = ch

        self.head = nn.Sequential(
            nn.Linear(2 * channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    @staticmethod
    def _stats_pool(h: torch.Tensor) -> torch.Tensor:
        mean = h.mean(dim=2)
        std = h.std(dim=2)
        return torch.cat([mean, std], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.stack([x.real, x.imag], dim=1)
        for block in self.features:
            h = block(h)
        pooled = self._stats_pool(h)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> dict:
    total = real_equiv = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        real_equiv += p.numel() * (2 if p.is_complex() else 1)
    return {"count": total, "real": real_equiv}


def build_model(name: str, config) -> nn.Module:
    if name == "complex":
        return ComplexModClassifier(
            n_classes=config.n_classes,
            channels=config.complex_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            hidden_dim=config.hidden_dim,
        )
    if name == "complex_moment":
        return ComplexMomentClassifier(
            n_classes=config.n_classes,
            channels=config.complex_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            hidden_dim=config.moment_hidden_dim,
            moment_orders=config.moment_orders,
        )
    if name == "real":
        return RealModClassifier(
            n_classes=config.n_classes,
            channels=config.real_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            hidden_dim=config.hidden_dim,
        )
    raise ValueError(
        f"unknown model '{name}', expected 'complex', 'complex_moment', or 'real'"
    )
