"""Small real and complex CNNs for raw IQ classification."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RealEncoder(nn.Module):
    """A conventional CNN that treats I and Q as two real channels."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        channels = (24, 48, 64)
        layers = []
        in_channels = 2
        for out_channels, kernel in zip(channels, (7, 5, 5)):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, 2, kernel // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Linear(3 * channels[-1], embedding_dim)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = torch.stack((iq.real, iq.imag), dim=1)
        x = self.features(x)
        stats = torch.cat((x.mean(-1), x.std(-1, unbiased=False), x.amax(-1)), dim=1)
        return self.projection(stats)


class ComplexConv1d(nn.Module):
    """Native complex-valued convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        scale = 1 / math.sqrt(2 * in_channels * kernel)
        self.weight = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, kernel, dtype=torch.complex64)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        self.padding = kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight, self.bias, stride=2, padding=self.padding)


class ComplexBlock(nn.Module):
    """Complex convolution followed by magnitude-preserving normalization."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel)
        self.norm = nn.BatchNorm1d(2 * out_channels)
        self.threshold = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        pair = self.norm(torch.cat((z.real, z.imag), dim=1))
        real, imag = pair.chunk(2, dim=1)
        z = torch.complex(real, imag)
        magnitude = torch.abs(z)
        return F.relu(magnitude + self.threshold[None, :, None]) * z / (magnitude + 1e-8)


class ComplexEncoder(nn.Module):
    """A capacity-matched CNN that keeps activations complex-valued."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        channels = (20, 30, 40)
        self.features = nn.Sequential(
            ComplexBlock(1, channels[0], 7),
            ComplexBlock(channels[0], channels[1], 5),
            ComplexBlock(channels[1], channels[2], 5),
        )
        self.projection = nn.Linear(6 * channels[-1], embedding_dim)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        z = self.features(iq[:, None])
        magnitude = torch.abs(z)
        stats = torch.cat(
            (
                z.real.mean(-1),
                z.real.std(-1, unbiased=False),
                z.imag.mean(-1),
                z.imag.std(-1, unbiased=False),
                magnitude.mean(-1),
                magnitude.amax(-1),
            ),
            dim=1,
        )
        return self.projection(stats)


class EmitterClassifier(nn.Module):
    """Attach a linear emitter classifier to an encoder."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_emitters: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, num_emitters)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(iq))


def real_scalar_parameters(model: nn.Module) -> int:
    """Count real scalars, counting a complex parameter as two scalars."""
    return sum(parameter.numel() * (2 if parameter.is_complex() else 1) for parameter in model.parameters())
