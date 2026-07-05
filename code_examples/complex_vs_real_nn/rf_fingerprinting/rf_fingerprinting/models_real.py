"""Real-valued encoder for RF fingerprinting."""

from __future__ import annotations

import torch
import torch.nn as nn


class RealEncoder1D(nn.Module):
    """Real-valued 1D convolutional encoder over I/Q channels.

    Input is complex IQ of shape ``[B, T]`` and is converted to a real tensor
    with 2 channels ``[I, Q]`` before convolution.
    """

    def __init__(self, embed_dim: int = 64, channels: tuple[int, int, int] = (32, 64, 64)):
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
        z = self.fc(h)
        return z
