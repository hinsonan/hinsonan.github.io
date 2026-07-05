"""Loss functions for contrastive and supervised RF tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute SimCLR NT-Xent loss.

    Args:
        z1: Projection vectors from view 1, shape ``[B, D]``.
        z2: Projection vectors from view 2, shape ``[B, D]``.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    n = z.shape[0]

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    batch = z1.shape[0]
    labels = torch.cat(
        [torch.arange(batch, 2 * batch, device=z.device), torch.arange(0, batch, device=z.device)]
    )

    return F.cross_entropy(sim, labels)
