"""Real and complex encoders, projection heads, and classifier wrappers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_encoder_options(
    channels: tuple[int, int, int],
    kernel_sizes: tuple[int, int, int],
    strides: tuple[int, int, int],
    pooling: str,
) -> None:
    """Validate options shared by the real and complex encoders."""
    if len(channels) != 3 or any(c <= 0 for c in channels):
        raise ValueError("channels must contain three positive widths")
    if len(kernel_sizes) != 3 or any(k <= 0 for k in kernel_sizes):
        raise ValueError("kernel_sizes must contain three positive values")
    if len(strides) != 3 or any(s <= 0 for s in strides):
        raise ValueError("strides must contain three positive values")
    if pooling not in {"avg", "max", "stats"}:
        raise ValueError("pooling must be 'avg', 'max', or 'stats'")


class RealEncoder1D(nn.Module):
    """Real-valued 1D convolutional encoder over I/Q channels.

    Input is complex IQ of shape ``[B, T]`` and is converted to a real
    tensor with two channels ``[I, Q]`` before convolution.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        channels: tuple[int, int, int] = (32, 64, 64),
        kernel_sizes: tuple[int, int, int] = (7, 5, 5),
        strides: tuple[int, int, int] = (2, 2, 2),
        pooling: str = "avg",
    ):
        """Initialize a real-valued RF encoder.

        Args:
            embed_dim: Output embedding dimension.
            channels: Convolution channel widths.
            kernel_sizes: Kernel sizes for the three convolution blocks.
            strides: Strides for the three convolution blocks.
            pooling: Global pooling mode. ``avg`` is the matched default;
                ``max`` and ``stats`` are also available for ablations.
        """
        super().__init__()
        _validate_encoder_options(channels, kernel_sizes, strides, pooling)
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        self.pooling = pooling
        self.net = nn.Sequential(
            nn.Conv1d(2, c1, kernel_size=k1, stride=s1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, stride=s2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, stride=s3, padding=k3 // 2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )
        pooled_dim = c3 if pooling != "stats" else 2 * c3
        self.fc = nn.Linear(pooled_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of IQ waveforms.

        Args:
            x: Complex tensor with shape ``[B, T]``.

        Returns:
            Embeddings with shape ``[B, embed_dim]``.
        """
        xr = torch.stack([x.real, x.imag], dim=1)
        h = self.net(xr)
        if self.pooling == "avg":
            pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        elif self.pooling == "max":
            pooled = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        else:
            pooled = torch.cat([h.mean(dim=2), h.std(dim=2, unbiased=False)], dim=1)
        return self.fc(pooled)


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
        """Compute a stable inverse square root of 2x2 symmetric matrices.

        Args:
            cov: Tensor of shape ``[..., 2, 2]`` containing symmetric PSD matrices.
            eps: Stabilizer added to the eigenvalues.

        Returns:
            Tensor of the same shape as ``cov`` with V^{-1/2} per matrix.

        The closed-form 2x2 expression avoids ``eigh`` eigenvector gradients,
        which are undefined when the covariance has repeated eigenvalues.
        """
        eye = torch.eye(2, dtype=cov.dtype, device=cov.device)
        regularized = cov + eps * eye
        a = regularized[..., 0, 0]
        b = regularized[..., 0, 1]
        c = regularized[..., 1, 1]
        determinant = torch.clamp(a * c - b.square(), min=eps * eps)
        root_det = torch.sqrt(determinant)
        trace_term = torch.sqrt(torch.clamp(a + c + 2 * root_det, min=eps))

        # A^-1/2 = sqrt(tr(A)+2sqrt(det(A))) * (A+sqrt(det(A))I)^-1.
        shifted_a = a + root_det
        shifted_c = c + root_det
        shifted_det = torch.clamp(shifted_a * shifted_c - b.square(), min=eps * eps)
        inverse = torch.stack(
            [
                torch.stack([shifted_c, -b], dim=-1),
                torch.stack([-b, shifted_a], dim=-1),
            ],
            dim=-2,
        ) / shifted_det.unsqueeze(-1).unsqueeze(-1)
        return trace_term.unsqueeze(-1).unsqueeze(-1) * inverse

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
    """Complex-valued 1D encoder with configurable matched pooling."""

    def __init__(
        self,
        embed_dim: int = 64,
        channels: tuple[int, int, int] = (32, 64, 64),
        moment_orders: tuple[int, ...] = (2, 4),
        kernel_sizes: tuple[int, int, int] = (7, 5, 5),
        strides: tuple[int, int, int] = (2, 2, 2),
        pooling: str = "avg",
    ):
        """Initialize a complex-valued RF encoder.

        Args:
            embed_dim: Output embedding dimension.
            channels: Convolution channel widths.
            moment_orders: Circular moment orders used in pooled statistics.
            kernel_sizes: Kernel sizes for the three convolution blocks.
            strides: Strides for the three convolution blocks.
            pooling: ``avg`` pools magnitude like the real encoder, ``max``
                pools magnitude with max pooling, and ``stats`` enables the
                original phase-aware statistics pooling.
        """
        super().__init__()
        _validate_encoder_options(channels, kernel_sizes, strides, pooling)
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        self.moment_orders = moment_orders
        self.pooling = pooling
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ComplexConv1d(1, c1, k1, stride=s1, padding=k1 // 2),
                    ComplexBatchNorm1d(c1),
                    ModReLU(),
                ),
                nn.Sequential(
                    ComplexConv1d(c1, c2, k2, stride=s2, padding=k2 // 2),
                    ComplexBatchNorm1d(c2),
                    ModReLU(),
                ),
                nn.Sequential(
                    ComplexConv1d(c2, c3, k3, stride=s3, padding=k3 // 2),
                    ComplexBatchNorm1d(c3),
                    ModReLU(),
                ),
            ]
        )
        if pooling == "stats":
            pooled_dim = (6 + len(moment_orders)) * c3
            self.head = nn.Sequential(
                nn.Linear(pooled_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.head = nn.Linear(c3, embed_dim)

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
        if self.pooling == "avg":
            pooled = F.adaptive_avg_pool1d(torch.abs(h), 1).squeeze(-1)
            return self.head(pooled)
        if self.pooling == "max":
            pooled = F.adaptive_max_pool1d(torch.abs(h), 1).squeeze(-1)
            return self.head(pooled)

        mag = torch.abs(h)
        unit = h / (mag + 1e-8)
        stats = [
            mag.mean(dim=2),
            mag.std(dim=2, unbiased=False),
            h.real.mean(dim=2),
            h.real.std(dim=2, unbiased=False),
            h.imag.mean(dim=2),
            h.imag.std(dim=2, unbiased=False),
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


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of scalar tensor elements in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def model_report(
    model: nn.Module,
    input_shape: tuple[int, int] = (1, 256),
) -> dict[str, int | tuple[int, ...]]:
    """Return parameter, real-scalar, MAC, and output-shape information.

    ``parameters`` counts tensor elements, while ``real_scalar_parameters``
    counts stored real and imaginary scalars. The latter is the appropriate
    capacity comparison between real and complex models. MACs are for one
    forward pass: ``conv_macs`` and ``linear_macs`` count real MAC-equivalents,
    with each complex convolution MAC counted as four real multiplications.
    Batch normalization, activations, pooling, and complex whitening are not
    included. The input is complex IQ, preserving the encoder interface.
    """
    if len(input_shape) != 2 or any(size <= 0 for size in input_shape):
        raise ValueError("input_shape must be (batch, sequence_length)")

    real_conv_macs = 0
    complex_conv_macs = 0
    linear_macs = 0
    hooks = []

    def record(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        nonlocal real_conv_macs, complex_conv_macs, linear_macs
        if isinstance(module, ComplexConv1d):
            batch, out_channels, length = output.shape
            complex_conv_macs += (
                batch
                * out_channels
                * length
                * module.weight.shape[1]
                * module.weight.shape[2]
                * 4
            )
        elif isinstance(module, nn.Conv1d):
            batch, out_channels, length = output.shape
            real_conv_macs += batch * out_channels * length * module.in_channels * module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            linear_macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (ComplexConv1d, nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(record))

    was_training = model.training
    try:
        model.eval()
        parameter = next(model.parameters())
        device = parameter.device
        if parameter.is_complex():
            input_dtype = parameter.dtype
        elif parameter.dtype == torch.float64:
            input_dtype = torch.complex128
        else:
            input_dtype = torch.complex64
        x = torch.zeros(input_shape, dtype=input_dtype, device=device)
        with torch.no_grad():
            output = model(x)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    real_scalar_parameters = sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
    )
    trainable_real_scalar_parameters = sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "real_scalar_parameters": real_scalar_parameters,
        "trainable_real_scalar_parameters": trainable_real_scalar_parameters,
        "real_conv_macs": real_conv_macs,
        "complex_conv_macs": complex_conv_macs,
        "conv_macs": real_conv_macs + complex_conv_macs,
        "linear_macs": linear_macs,
        "macs": real_conv_macs + complex_conv_macs + linear_macs,
        "output_shape": tuple(output.shape),
    }
