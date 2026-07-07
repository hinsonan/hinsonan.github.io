"""Complex-valued encoder for RF fingerprinting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """Initialize a complex convolution layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size.
            stride: Convolution stride.
            padding: Zero padding.
        """
        super().__init__()
        std = 1.0 / math.sqrt(2.0 * in_channels * kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, dtype=torch.complex64) * std
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex convolution."""
        return F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class ModReLU(nn.Module):
    """modReLU nonlinearity for complex activations."""

    def __init__(self):
        """Initialize learnable threshold."""
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply modReLU."""
        mag = torch.abs(z)
        return torch.relu(mag + self.b) * z / (mag + 1e-8)


class ComplexBatchNorm1d(nn.Module):
    """Batch-normalize real and imaginary components independently."""

    def __init__(self, num_features: int):
        """Initialize paired batch norms.

        Args:
            num_features: Number of complex channels.
        """
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply batch norm to real and imaginary parts.

        Args:
            z: Complex activation tensor with shape ``[B, C, T]``.

        Returns:
            Complex tensor with independently normalized components.
        """
        return torch.complex(self.bn_real(z.real), self.bn_imag(z.imag))


class ComplexEncoder1D(nn.Module):
    """Complex-valued 1D encoder with normalized blocks and rich pooling."""

    def __init__(
        self,
        embed_dim: int = 64,
        channels: tuple[int, int, int] = (24, 48, 48),
        moment_orders: tuple[int, ...] = (2, 4),
    ):
        """Initialize a complex-valued RF encoder.

        Args:
            embed_dim: Output embedding dimension.
            channels: Convolution channel widths.
            moment_orders: Circular moment orders used in pooled statistics.
        """
        super().__init__()
        c1, c2, c3 = channels
        self.moment_orders = moment_orders
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ComplexConv1d(1, c1, 7, stride=2, padding=3),
                    ComplexBatchNorm1d(c1),
                    ModReLU(),
                ),
                nn.Sequential(
                    ComplexConv1d(c1, c2, 5, stride=2, padding=2),
                    ComplexBatchNorm1d(c2),
                    ModReLU(),
                ),
                nn.Sequential(
                    ComplexConv1d(c2, c3, 5, stride=2, padding=2),
                    ComplexBatchNorm1d(c3),
                    ModReLU(),
                ),
            ]
        )
        n_stats = 6 + len(moment_orders)
        self.fc = nn.Linear(n_stats * c3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of IQ waveforms.

        Args:
            x: Complex tensor with shape ``[B, T]``.

        Returns:
            Embeddings with shape ``[B, embed_dim]``.
        """
        h = x.unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        mag = torch.abs(h)
        unit = h / (mag + 1e-8)
        stats = [
            mag.mean(dim=2),
            mag.std(dim=2),
            h.real.mean(dim=2),
            h.real.std(dim=2),
            h.imag.mean(dim=2),
            h.imag.std(dim=2),
        ]
        for order in self.moment_orders:
            stats.append(torch.abs(torch.mean(unit ** order, dim=2)))
        pooled = torch.cat(stats, dim=1)
        return self.fc(pooled)
