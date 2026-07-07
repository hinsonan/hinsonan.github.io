"""SimCLR-style pretraining routines for RF encoders."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .config import RFConfig
from .heads import ProjectionHead
from .losses import nt_xent_loss


def pretrain_simclr(
    encoder: torch.nn.Module,
    dataset,
    cfg: RFConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Run short SimCLR pretraining.

    Args:
        encoder: Feature encoder.
        dataset: Two-view dataset yielding ``(v1, v2)``.
        cfg: Runtime configuration.
        device: Compute device.

    Returns:
        History dictionary with epoch losses.
    """
    encoder = encoder.to(device)
    proj = ProjectionHead(cfg.embed_dim, cfg.embed_dim).to(device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.pretrain_epochs, 1), eta_min=cfg.lr * 0.1
    )

    history = {"loss": []}
    for _ in range(cfg.pretrain_epochs):
        encoder.train()
        proj.train()
        running = 0.0
        count = 0
        for v1, v2 in loader:
            v1 = v1.to(device)
            v2 = v2.to(device)
            z1 = proj(encoder(v1))
            z2 = proj(encoder(v2))
            loss = nt_xent_loss(z1, z2, temperature=cfg.temperature)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            count += 1
        scheduler.step()
        history["loss"].append(running / max(count, 1))
    return history
