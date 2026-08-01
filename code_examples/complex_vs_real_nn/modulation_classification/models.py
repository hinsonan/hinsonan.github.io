"""Model definitions for the AMC experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """1D convolution over complex-valued signals.

    Operates natively on complex64/bfloat16 tensors via PyTorch's
    complex-to-complex convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        """Initialise the complex convolution layer.

        Weights are initialised with a complex-valued scaled normal
        distribution following the Glorot-inspired heuristic.

        Args:
            in_channels: Number of complex input channels.
            out_channels: Number of complex output channels.
            kernel_size: Length of the 1D convolution kernel.
            stride: Convolution stride.
            padding: Zero-padding applied to both sides of the input.
            bias: Whether to include a learnable complex bias.
        """
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
        """Apply the complex convolution.

        Args:
            x: Input tensor of shape ``(batch, in_channels, length)``.

        Returns:
            Convolved output of shape ``(batch, out_channels, out_length)``.
        """
        return F.conv1d(
            x, self.weight, self.bias, stride=self.stride, padding=self.padding
        )


class modReLU(nn.Module):
    """Modulus ReLU activation for complex-valued activations.

    Applies ``relu(|z| + b) * z / (|z| + eps)``, keeping the phase
    while gating the magnitude with a learnable threshold ``b``.
    """

    def __init__(self):
        """Initialise with a learnable threshold parameter (scalar)."""
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply modReLU non-linearity.

        Args:
            z: Complex input tensor.

        Returns:
            Activated tensor of the same shape and dtype.
        """
        mag = torch.abs(z)
        return torch.relu(mag + self.b) * z / (mag + 1e-8)


class ComplexModClassifier(nn.Module):
    """Modulation classifier built from complex-valued convolutions.

    Architecture is a stack of three ``ComplexConv1d + modReLU`` blocks
    followed by a magnitude-pooling head that extracts mean and standard
    deviation of each channel's magnitude over time.
    """

    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (24, 48, 48),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
    ):
        """Initialise the complex modulation classifier.

        Args:
            n_classes: Number of output modulation classes.
            channels: Channel count for each of the three conv blocks.
            kernel_size: Convolution kernel length.
            stride: Convolution stride (same for all blocks).
            hidden_dim: Width of the MLP head's hidden layer.
        """
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
        """Pool over time with magnitude mean and standard deviation.

        Args:
            h: Complex activations of shape ``(batch, channels, time)``.

        Returns:
            Concatenated [mean, std] of shape ``(batch, 2 * channels)``.
        """
        mag = torch.abs(h)
        mean = mag.mean(dim=2)
        std = mag.std(dim=2)
        return torch.cat([mean, std], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of complex baseband bursts.

        Args:
            x: Complex input tensor of shape ``(batch, burst_len)``.

        Returns:
            Raw class logits of shape ``(batch, n_classes)``.
        """
        h = x.unsqueeze(1)
        for block in self.features:
            h = block(h)
        pooled = self._magnitude_pool(h)
        return self.head(pooled)


class ComplexMomentClassifier(nn.Module):
    """Modulation classifier with complex convolutions and higher-order
    moment pooling.

    Extends the basic :class:`ComplexModClassifier` by additionally
    computing higher-order circular moments of the final complex
    activations, providing richer rotation-invariant statistics.
    """

    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (24, 48, 48),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
        moment_orders: tuple = (2, 4, 8),
    ):
        """Initialise the complex moment classifier.

        Args:
            n_classes: Number of output modulation classes.
            channels: Channel count for each of the three conv blocks.
            kernel_size: Convolution kernel length.
            stride: Convolution stride (same for all blocks).
            hidden_dim: Width of the MLP head's hidden layer.
            moment_orders: Orders of circular moments to compute
                on the normalised complex activations.
        """
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
        """Pool over time with magnitude stats and circular moments.

        Computes mean, std of magnitude, and the absolute value of
        the mean of ``(h / |h|)^order`` for each specified order.

        Args:
            h: Complex activations of shape ``(batch, channels, time)``.

        Returns:
            Concatenated statistics of shape
            ``(batch, (2 + len(moment_orders)) * channels)``.
        """
        mag = torch.abs(h)
        stats = [mag.mean(dim=2), mag.std(dim=2)]

        unit = h / (mag + 1e-8)
        for order in self.moment_orders:
            stats.append(torch.abs(torch.mean(unit ** order, dim=2)))
        return torch.cat(stats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of complex baseband bursts.

        Args:
            x: Complex input tensor of shape ``(batch, burst_len)``.

        Returns:
            Raw class logits of shape ``(batch, n_classes)``.
        """
        h = x.unsqueeze(1)
        for block in self.features:
            h = block(h)
        pooled = self._moment_pool(h)
        return self.head(pooled)


class RealModClassifier(nn.Module):
    """Modulation classifier using real-valued convolutions on
    I/Q split into 2 input channels.

    The complex input is stacked as ``[real, imag]`` along the channel
    dimension, then processed by standard ``Conv1d + BN + ReLU`` blocks.
    """

    def __init__(
        self,
        n_classes: int = 4,
        channels: tuple = (32, 64, 64),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
    ):
        """Initialise the real modulation classifier.

        Args:
            n_classes: Number of output modulation classes.
            channels: Channel count for each of the three conv blocks.
            kernel_size: Convolution kernel length.
            stride: Convolution stride (same for all blocks).
            hidden_dim: Width of the MLP head's hidden layer.
        """
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
        """Pool over time with mean and standard deviation.

        Args:
            h: Real activations of shape ``(batch, channels, time)``.

        Returns:
            Concatenated [mean, std] of shape ``(batch, 2 * channels)``.
        """
        mean = h.mean(dim=2)
        std = h.std(dim=2)
        return torch.cat([mean, std], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of complex baseband bursts.

        The complex input is first split into I/Q channels as a real
        tensor of shape ``(batch, 2, burst_len)``.

        Args:
            x: Complex input tensor of shape ``(batch, burst_len)``.

        Returns:
            Raw class logits of shape ``(batch, n_classes)``.
        """
        h = torch.stack([x.real, x.imag], dim=1)
        for block in self.features:
            h = block(h)
        pooled = self._stats_pool(h)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> dict:
    """Count trainable parameters and their real-valued equivalent.

    Complex parameters are counted as 2 real params (I + Q components).

    Args:
        model: A PyTorch model.

    Returns:
        Dictionary with keys:
            'count': raw parameter count (complex params count as 1),
            'real': real-valued equivalent parameter count.
    """
    total = real_equiv = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        real_equiv += p.numel() * (2 if p.is_complex() else 1)
    return {"count": total, "real": real_equiv}


def build_model(name: str, config) -> nn.Module:
    """Build a model by name using the configuration's hyper-parameters.

    Args:
        name: Model key — ``'complex'``, ``'complex_moment'``, or ``'real'``.
        config: A :class:`ModClassConfig` instance (or any object with the
            relevant attributes).

    Returns:
        Initialised model (on CPU).

    Raises:
        ValueError: If ``name`` is not recognised.
    """
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
