"""Real and complex encoders, projection heads, and classifier wrappers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealEncoder1D(nn.Module):
    """Real-valued 1D convolutional encoder over I/Q channels.

    Input is complex IQ of shape ``[B, T]`` and is converted to a real
    tensor with two channels ``[I, Q]`` before convolution.
    """

    def __init__(
        self, embed_dim: int = 64, channels: tuple[int, int, int] = (32, 64, 64)
    ):
        """Initialize a real-valued RF encoder.

        Args:
            embed_dim: Output embedding dimension.
            channels: Convolution channel widths.
        """
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv1d(2, c1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(c3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of IQ waveforms.

        Args:
            x: Complex tensor with shape ``[B, T]``.

        Returns:
            Embeddings with shape ``[B, embed_dim]``.
        """
        xr = torch.stack([x.real, x.imag], dim=1)
        h = self.net(xr).squeeze(-1)
        return self.fc(h)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
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
        """Apply complex convolution.

        Args:
            x: Complex input tensor.

        Returns:
            Convolved complex tensor.
        """
        return F.conv1d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding
        )


class ModReLU(nn.Module):
    """modReLU nonlinearity for complex activations."""

    def __init__(self):
        """Initialize learnable threshold."""
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply modReLU.

        Args:
            z: Complex input tensor.

        Returns:
            Activated complex tensor.
        """
        mag = torch.abs(z)
        return torch.relu(mag + self.b) * z / (mag + 1e-8)


class ComplexBatchNorm1d(nn.Module):
    """Complex batch normalization with covariance whitening.

    Implements the "Complex Batch Normalization" variant from
    Trabelsi et al. 2018 ("Deep Complex Networks") that whitens the
    2D real-valued distribution ``[Re(z), Im(z)]`` per channel using
    the inverse square root of its 2x2 covariance matrix. The result
    is rotation-equivariant: a global phase rotation of the input
    rotates the mean and leaves the covariance unchanged, so the
    whitened output is also just rotated.

    Args:
        num_features: Number of complex channels.
        eps: Numerical stability for the covariance matrix.
        momentum: Running-statistics momentum.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """Initialize complex BN parameters and running statistics."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(
            torch.ones(num_features, dtype=torch.complex64)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_features, dtype=torch.complex64)
        )

        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.complex64)
        )
        self.register_buffer(
            "running_cov", torch.eye(2).expand(num_features, 2, 2).clone()
        )

    @staticmethod
    def _inv_sqrt(cov: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute the inverse square root of a batch of 2x2 symmetric matrices.

        Args:
            cov: Tensor of shape ``[..., 2, 2]`` containing symmetric PSD matrices.
            eps: Stabilizer added to the eigenvalues.

        Returns:
            Tensor of the same shape as ``cov`` with V^{-1/2} per matrix.
        """
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=eps)
        inv_sqrt = eigvecs @ torch.diag_embed(1.0 / torch.sqrt(eigvals)) @ eigvecs.transpose(-1, -2)
        return inv_sqrt

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply complex batch normalization.

        Args:
            z: Complex activation tensor with shape ``[B, C, T]``.

        Returns:
            Complex tensor of the same shape.
        """
        if self.training:
            mean = z.mean(dim=(0, 2))
            z_centered = z - mean.view(1, -1, 1)
            cov_rr = (z_centered.real ** 2).mean(dim=(0, 2))
            cov_ii = (z_centered.imag ** 2).mean(dim=(0, 2))
            cov_ri = (z_centered.real * z_centered.imag).mean(dim=(0, 2))
            cov = torch.stack(
                [
                    torch.stack([cov_rr, cov_ri], dim=-1),
                    torch.stack([cov_ri, cov_ii], dim=-1),
                ],
                dim=-2,
            )
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )
                self.running_cov = (
                    (1 - self.momentum) * self.running_cov + self.momentum * cov
                )
        else:
            mean = self.running_mean
            cov = self.running_cov

        inv_sqrt = self._inv_sqrt(cov, self.eps)

        z_r = z.real - mean.real.view(1, -1, 1)
        z_i = z.imag - mean.imag.view(1, -1, 1)
        z_2d = torch.stack([z_r, z_i], dim=-1)
        z_white = torch.einsum("bctj,cjk->bctk", z_2d, inv_sqrt)
        z_white_complex = torch.complex(z_white[..., 0], z_white[..., 1])

        return self.weight.view(1, -1, 1) * z_white_complex + self.bias.view(1, -1, 1)


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
        self.head = nn.Sequential(
            nn.Linear(n_stats * c3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

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
        return self.head(pooled)


class ProjectionHead(nn.Module):
    """MLP projection head used in SimCLR pretraining."""

    def __init__(self, in_dim: int, proj_dim: int = 64):
        """Initialize projection head.

        Args:
            in_dim: Input embedding dimension.
            proj_dim: Projection output dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings for contrastive loss."""
        return self.net(x)


class ClassifierHead(nn.Module):
    """Linear classifier head for probing and fine-tuning."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        """Initialize classifier head.

        Args:
            in_dim: Input embedding dimension.
            num_classes: Number of classes.
            dropout: Dropout probability before the linear layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from embeddings."""
        return self.fc(self.dropout(x))


class EncoderClassifier(nn.Module):
    """Wrap an encoder with a classifier head."""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        """Initialize model.

        Args:
            encoder: Feature encoder.
            embed_dim: Embedding dimension.
            num_classes: Number of classes.
            dropout: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = encoder
        self.head = ClassifierHead(embed_dim, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch."""
        return self.head(self.encoder(x))
