"""Training, probing, and evaluation routines for RF fingerprinting."""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import RFConfig
from models import EncoderClassifier, ProjectionHead


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
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


def pretrain_simclr(
    encoder: nn.Module,
    dataset,
    cfg: RFConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Run SimCLR pretraining on a two-view dataset.

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


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy over a dataloader.

    Args:
        model: Classifier model.
        loader: Validation/test loader.
        device: Compute device.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += int((pred == y).sum().item())
    return correct / max(total, 1)


def _build_optimizer(model: nn.Module, cfg: RFConfig) -> torch.optim.Optimizer:
    """Build an Adam optimizer with a lower LR for the encoder.

    Args:
        model: The encoder-classifier model.
        cfg: Runtime configuration.

    Returns:
        Configured Adam optimizer.
    """
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())

    param_groups = [
        {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_scale},
        {"params": head_params, "lr": cfg.lr},
    ]
    return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)


def finetune_classifier(
    encoder: nn.Module,
    train_dataset,
    val_dataset,
    num_classes: int,
    cfg: RFConfig,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, List[float]]]:
    """Fine-tune encoder + classifier head with supervised labels.

    Saves and returns the model with the highest validation accuracy.

    Args:
        encoder: Feature encoder.
        train_dataset: Supervised train dataset.
        val_dataset: Supervised validation dataset.
        num_classes: Number of device classes.
        cfg: Runtime config.
        device: Compute device.

    Returns:
        Tuple of ``(best_model, history)``.
    """
    model = EncoderClassifier(encoder, cfg.embed_dim, num_classes, dropout=cfg.dropout).to(device)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    opt = _build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.finetune_epochs, 1), eta_min=cfg.lr * 0.1
    )

    history: Dict[str, List[float]] = {"loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for _ in range(cfg.finetune_epochs):
        model.train()
        running = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            count += 1
        scheduler.step()

        val_acc = _accuracy(model, val_loader, device)
        history["loss"].append(running / max(count, 1))
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module, dataset, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from a supervised dataset.

    Args:
        encoder: Feature encoder.
        dataset: Dataset yielding ``(iq, label)``.
        device: Compute device.

    Returns:
        Tuple ``(embeddings, labels)`` as NumPy arrays.
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x).cpu().numpy()
        feats.append(z)
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run_linear_probe(
    encoder: nn.Module,
    train_dataset,
    test_dataset,
    device: torch.device,
    max_iter: int = 500,
) -> Dict[str, float]:
    """Train and evaluate a linear probe on frozen embeddings.

    Args:
        encoder: Frozen feature encoder.
        train_dataset: Supervised train dataset.
        test_dataset: Supervised test dataset.
        device: Compute device.
        max_iter: Max logistic regression iterations.

    Returns:
        Metrics dictionary.
    """
    x_train, y_train = extract_embeddings(encoder, train_dataset, device)
    x_test, y_test = extract_embeddings(encoder, test_dataset, device)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=max_iter, n_jobs=1, class_weight="balanced")
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {"probe_acc": float(accuracy_score(y_test, pred))}


@torch.no_grad()
def collect_logits(
    model: nn.Module, dataset, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Collect logits and labels from a dataset.

    Args:
        model: Classifier model.
        dataset: Dataset yielding ``(iq, label)``.
        device: Compute device.

    Returns:
        Tuple ``(logits, labels)`` as NumPy arrays.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    logits = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        logits.append(out)
        labels.append(y.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute scalar metrics from logits and labels.

    Args:
        logits: Logits array ``[N, C]``.
        labels: True labels array ``[N]``.

    Returns:
        Dictionary with accuracy and macro F1.
    """
    pred = np.argmax(logits, axis=1)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
    }


def max_softmax_scores(logits: np.ndarray) -> np.ndarray:
    """Compute max-softmax confidence scores.

    Args:
        logits: Logits array ``[N, C]``.

    Returns:
        Confidence scores in ``[0, 1]``.
    """
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    return np.max(prob, axis=1)


def open_set_auc(id_logits: np.ndarray, ood_logits: np.ndarray) -> Dict[str, float]:
    """Compute AUROC for in-distribution vs out-of-distribution detection.

    Args:
        id_logits: In-distribution logits.
        ood_logits: Out-of-distribution logits.

    Returns:
        Dictionary with AUROC based on max-softmax confidence.
    """
    id_score = max_softmax_scores(id_logits)
    ood_score = max_softmax_scores(ood_logits)
    y = np.concatenate([np.ones_like(id_score), np.zeros_like(ood_score)])
    s = np.concatenate([id_score, ood_score])
    return {"open_set_auroc": float(roc_auc_score(y, s))}
