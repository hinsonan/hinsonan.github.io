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
    """Native complex convolution with optional conjugate processing."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, widely_linear: bool = False):
        super().__init__()
        scale = 1 / math.sqrt(2 * in_channels * kernel)
        self.weight = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, kernel, dtype=torch.complex64)
        )
        self.conjugate_weight = (
            nn.Parameter(scale * torch.randn(out_channels, in_channels, kernel, dtype=torch.complex64))
            if widely_linear
            else None
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        self.padding = kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.conv1d(x, self.weight, self.bias, stride=2, padding=self.padding)
        if self.conjugate_weight is not None:
            output = output + F.conv1d(
                x.conj(), self.conjugate_weight, stride=2, padding=self.padding
            )
        return output


class ComplexBatchNorm1d(nn.Module):
    """Whiten each complex channel using its 2-by-2 I/Q covariance."""

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(channels, dtype=torch.complex64))
        self.register_buffer("running_covariance", torch.eye(2).repeat(channels, 1, 1))
        self.weight_rr = nn.Parameter(torch.ones(channels))
        self.weight_ii = nn.Parameter(torch.ones(channels))
        self.weight_ri = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.complex64))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = z.mean(dim=(0, 2))
            centered = z - mean[None, :, None]
            pair = torch.stack((centered.real, centered.imag), dim=-1)
            covariance = torch.einsum("bcli,bclj->cij", pair, pair) / (z.shape[0] * z.shape[2])
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_covariance.lerp_(covariance.detach(), self.momentum)
        else:
            mean = self.running_mean
            covariance = self.running_covariance
            centered = z - mean[None, :, None]
            pair = torch.stack((centered.real, centered.imag), dim=-1)

        covariance = covariance + self.eps * torch.eye(2, device=z.device)[None]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_sqrt = eigenvectors @ torch.diag_embed(eigenvalues.rsqrt()) @ eigenvectors.transpose(-2, -1)
        pair = torch.einsum("cij,bclj->bcli", inverse_sqrt, pair)
        affine = torch.stack(
            (
                torch.stack((self.weight_rr, self.weight_ri), dim=-1),
                torch.stack((self.weight_ri, self.weight_ii), dim=-1),
            ),
            dim=-2,
        )
        pair = torch.einsum("cij,bclj->bcli", affine, pair)
        return torch.complex(pair[..., 0], pair[..., 1]) + self.bias[None, :, None]


class ComplexBlock(nn.Module):
    """Complex convolution followed by magnitude-preserving normalization."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, widely_linear: bool = False
    ):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel, widely_linear)
        self.norm = ComplexBatchNorm1d(out_channels)
        self.threshold = nn.Parameter(torch.full((out_channels,), -0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.conv(x))
        magnitude = torch.abs(z)
        return F.relu(magnitude + self.threshold[None, :, None]) * z / (magnitude + 1e-8)


class ComplexEncoder(nn.Module):
    """A capacity-matched CNN that keeps activations complex-valued."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        channels = (20, 30, 40)
        self.features = nn.Sequential(
            ComplexBlock(1, channels[0], 7, widely_linear=True),
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
