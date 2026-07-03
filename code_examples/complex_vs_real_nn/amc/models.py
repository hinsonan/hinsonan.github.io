"""Model definitions for the AMC experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
