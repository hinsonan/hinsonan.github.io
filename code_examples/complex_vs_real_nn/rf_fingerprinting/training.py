"""Training, probing, and evaluation routines for RF fingerprinting."""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:  # Support both ``import training`` and package imports.
    from .config import RFConfig
    from .datasets import augment_iq
    from .models import EncoderClassifier, ProjectionHead
except ImportError:  # pragma: no cover - exercised by notebook-style imports.
    from config import RFConfig
    from datasets import augment_iq
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


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    Pulls same-label samples together and pushes different-label samples
    apart using cosine similarity in projection/embedding space.
    """
    features = F.normalize(features, dim=1)
    n = features.shape[0]
    device = features.device

    sim = torch.matmul(features, features.T) / temperature
    self_mask = torch.eye(n, device=device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float("-inf"))

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~self_mask

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim)
    exp = exp.masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    mean_log_prob_pos = log_prob.sum(dim=1) / pos_count.clamp(min=1)
    loss = -mean_log_prob_pos[valid]
    return loss.mean() if valid.any() else features.new_zeros(())


def _infer_encoder_dim(
    encoder: nn.Module, dataset=None, device: Optional[torch.device] = None
) -> int:
    """Infer an encoder output width, probing one dataset item only if needed."""
    for name in ("embed_dim", "output_dim"):
        value = getattr(encoder, name, None)
        if isinstance(value, int) and value > 0:
            return value
    for module in reversed(list(encoder.modules())):
        if isinstance(module, nn.Linear) and module.out_features > 0:
            return module.out_features
    if dataset is None or len(dataset) == 0:
        raise ValueError("cannot infer encoder output dimension; pass an explicit dimension")
    sample = dataset[0]
    sample = sample[0] if isinstance(sample, (tuple, list)) else sample
    target_device = device or next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        output = encoder(sample.unsqueeze(0).to(target_device))
    if was_training:
        encoder.train()
    if output.ndim != 2 or output.shape[1] <= 0:
        raise ValueError("encoder must return a non-empty [batch, dimension] tensor")
    return int(output.shape[1])


def pretrain_simclr(
    encoder: nn.Module,
    dataset,
    cfg: RFConfig,
    device: torch.device,
    seed: Optional[int] = None,
    num_workers: int = 0,
    loader_generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    deterministic: bool = False,
    projection_dim: Optional[int] = None,
    supcon_weight: float = 0.0,
) -> Dict[str, List[float]]:
    """Run SimCLR (optionally with SupCon) pretraining.

    When ``supcon_weight > 0`` the dataset must yield three tensors
    ``(v1, v2, label)``. The total loss is::

        loss = nt_xent(z1, z2) + supcon_weight * supcon([z1;z2], [y;y])
    """
    effective_seed = cfg.seed if seed is None else seed
    seed_everything(effective_seed, deterministic=deterministic)
    encoder = encoder.to(device)
    if len(dataset) == 0:
        raise ValueError("SimCLR dataset must contain at least one sample")
    encoder_dim = _infer_encoder_dim(encoder, dataset, device)
    projection_dim = encoder_dim if projection_dim is None else projection_dim
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    proj = ProjectionHead(encoder_dim, projection_dim).to(device)
    generator = loader_generator
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(effective_seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn if worker_init_fn is not None else (_seed_worker if num_workers else None),
    )

    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.pretrain_epochs, 1), eta_min=cfg.lr * 0.1
    )

    use_supcon = supcon_weight > 0.0
    history: Dict[str, List[float]] = {"loss": [], "loss_nt_xent": [], "loss_supcon": []} if use_supcon else {"loss": []}
    for epoch in range(cfg.pretrain_epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        encoder.train()
        proj.train()
        running = running_ntx = running_sup = 0.0
        count = 0
        for batch in loader:
            if use_supcon:
                v1, v2, y = (t.to(device) for t in batch)
            else:
                v1, v2 = (t.to(device) for t in batch)
                y = None
            z1 = proj(encoder(v1))
            z2 = proj(encoder(v2))
            loss_ntx = nt_xent_loss(z1, z2, temperature=cfg.temperature)
            if use_supcon:
                feats = torch.cat([z1, z2], dim=0)
                labels = torch.cat([y, y], dim=0)
                loss_sup = supervised_contrastive_loss(feats, labels, temperature=cfg.temperature)
                loss = loss_ntx + supcon_weight * loss_sup
                running_sup += float(loss_sup.item())
            else:
                loss = loss_ntx
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            running_ntx += float(loss_ntx.item())
            count += 1
        scheduler.step()
        history["loss"].append(running / max(count, 1))
        if use_supcon:
            history["loss_nt_xent"].append(running_ntx / max(count, 1))
            history["loss_supcon"].append(running_sup / max(count, 1))
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
    supcon_lambda: float = 0.0,
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
        supcon_lambda: Weight on supervised contrastive loss applied to
            encoder embeddings when > 0.

    Returns:
        Tuple of ``(best_model, history)``.
    """
    if len(train_dataset) == 0:
        raise ValueError("training dataset must contain at least one sample")
    encoder_dim = _infer_encoder_dim(encoder, train_dataset, device)
    model = EncoderClassifier(encoder, encoder_dim, num_classes, dropout=cfg.dropout).to(device)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    opt = _build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.finetune_epochs, 1), eta_min=cfg.lr * 0.1
    )

    use_supcon = supcon_lambda > 0.0
    history: Dict[str, List[float]] = (
        {"loss": [], "loss_ce": [], "loss_supcon": [], "val_acc": []}
        if use_supcon
        else {"loss": [], "val_acc": []}
    )
    best_val = -1.0
    best_state = None

    for epoch in range(cfg.finetune_epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        running = running_ce = running_sup = 0.0
        count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            emb = model.encoder(x)
            logits = model.head(emb)
            loss_ce = criterion(logits, y)
            loss = loss_ce
            if use_supcon:
                loss_sup = supervised_contrastive_loss(emb, y, temperature=cfg.temperature)
                loss = loss_ce + supcon_lambda * loss_sup
                running_sup += float(loss_sup.item())
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss.item())
            running_ce += float(loss_ce.item())
            count += 1
        scheduler.step()

        val_acc = _accuracy(model, val_loader, device)
        history["loss"].append(running / max(count, 1))
        if use_supcon:
            history["loss_ce"].append(running_ce / max(count, 1))
            history["loss_supcon"].append(running_sup / max(count, 1))
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def outlier_exposure_finetune(
    model: nn.Module,
    train_dataset,
    unknown_dataset,
    cfg: RFConfig,
    device: torch.device,
    oe_weight: float = 1.0,
    epochs: int = 2,
    lr_scale: float = 1.0,
    encoder_lr_scale: float = 0.0,
) -> nn.Module:
    """Fine-tune the classifier head for open-set scoring with outlier exposure.

    Keeps the encoder frozen and optimizes cross-entropy on known samples
    plus a uniform-softmax objective on unlabeled unknown samples (KL from
    the uniform distribution to the predicted softmax). Closed-set training
    otherwise makes logits maximally confident near known class centers --
    exactly where look-alike unknown devices land -- which drives
    max-softmax/energy AUROC below chance. When ``encoder_lr_scale > 0``
    the encoder is also adapted at a reduced LR, which reshapes the
    embedding space around the calibration unknowns instead of only
    recalibrating the head. The returned copy is intended for logit-based
    open-set scores only; the input model is left untouched.

    Args:
        model: Trained :class:`EncoderClassifier`.
        train_dataset: Labeled known-device dataset.
        unknown_dataset: Unlabeled unknown-device dataset (calibration split).
        cfg: Runtime config.
        device: Compute device.
        oe_weight: Weight on the uniform-softmax objective.
        epochs: Fine-tune epochs over the paired loaders.
        lr_scale: Multiplier on ``cfg.lr`` for the head optimizer.
        encoder_lr_scale: Additional multiplier on the head LR for the
            encoder parameter group. Zero (default) freezes the encoder.

    Returns:
        A new model with a calibrated head for logit-based open-set scoring.
    """
    if len(train_dataset) == 0 or len(unknown_dataset) == 0:
        raise ValueError("outlier exposure requires non-empty known and unknown datasets")
    if oe_weight < 0:
        raise ValueError("oe_weight must be non-negative")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if encoder_lr_scale < 0:
        raise ValueError("encoder_lr_scale must be non-negative")
    oe_model = copy.deepcopy(model).to(device)
    tune_encoder = encoder_lr_scale > 0
    if tune_encoder:
        opt = torch.optim.Adam(
            [
                {"params": oe_model.encoder.parameters(), "lr": cfg.lr * lr_scale * encoder_lr_scale},
                {"params": oe_model.head.parameters(), "lr": cfg.lr * lr_scale},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        for param in oe_model.encoder.parameters():
            param.requires_grad = False
        opt = torch.optim.Adam(oe_model.head.parameters(), lr=cfg.lr * lr_scale,
                               weight_decay=cfg.weight_decay)
    known_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    unknown_loader = DataLoader(unknown_dataset, batch_size=cfg.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        if tune_encoder:
            oe_model.train()
        else:
            oe_model.head.train()
            oe_model.encoder.eval()
        for (x_known, y_known), (x_unknown, _) in zip(known_loader, unknown_loader):
            x_known = x_known.to(device)
            y_known = y_known.to(device)
            x_unknown = x_unknown.to(device)
            known_logits = oe_model(x_known)
            unknown_logits = oe_model(x_unknown)
            log_probs = F.log_softmax(unknown_logits, dim=1)
            uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[1])
            loss = criterion(known_logits, y_known) + oe_weight * F.kl_div(
                log_probs, uniform, reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return oe_model


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


@torch.no_grad()
def _encode_array(encoder: nn.Module, iq: np.ndarray, device: torch.device,
                  batch_size: int = 256) -> np.ndarray:
    """Encode a complex IQ array in batches."""
    feats = []
    for start in range(0, iq.shape[0], batch_size):
        x = torch.from_numpy(iq[start:start + batch_size]).to(device)
        feats.append(encoder(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def extract_embeddings_tta(
    encoder: nn.Module,
    iq_samples: np.ndarray,
    cfg: RFConfig,
    device: torch.device,
    k_views: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Average L2-normalized embeddings over several augmented views.

    Test-time augmentation smooths noise realization effects, which
    stabilizes prototype-cosine and Mahalanobis open-set scores at low SNR.
    With ``k_views=1`` this reduces to encoding a single augmented view.
    """
    if k_views < 1:
        raise ValueError("k_views must be at least 1")
    iq = np.asarray(iq_samples, dtype=np.complex64)
    if iq.ndim != 2 or iq.shape[1] <= 0:
        raise ValueError("iq_samples must have shape [N, T] with T > 0")
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    encoder = encoder.to(device)
    encoder.eval()
    aug_kwargs = dict(
        noise_std=cfg.noise_std,
        phase_jitter_rad=cfg.phase_jitter_rad,
        time_shift=cfg.time_shift,
        cfo_jitter_rad=cfg.cfo_jitter_rad,
        amplitude_jitter=cfg.amplitude_jitter,
        aug_prob=cfg.aug_prob,
    )
    accumulated = None
    for _ in range(k_views):
        views = np.stack([augment_iq(x, rng=rng, **aug_kwargs) for x in iq])
        embeddings = _encode_array(encoder, views, device).astype(np.float64)
        embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        accumulated = embeddings if accumulated is None else accumulated + embeddings
    return accumulated / float(k_views)


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
    return {"open_set_auroc": evaluate_open_set(id_logits, ood_logits)["auroc"]}


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch, optionally enabling deterministic ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Select a usable accelerator, falling back when CUDA is incompatible.

    ``torch.cuda.is_available()`` can be true even when the installed PyTorch
    build has no kernel image for the local GPU.  A tiny synchronized operation
    catches that case before a model is moved to CUDA.
    """
    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.ones(1, device="cuda")
        (probe + 1).sum().item()
        torch.cuda.synchronize()
    except (RuntimeError, torch.AcceleratorError):
        return torch.device("cpu")
    return torch.device("cuda")


def _seed_worker(worker_id: int) -> None:
    """Initialize DataLoader workers from PyTorch's per-loader seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _validate_scores_labels(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_labels = np.asarray(labels).reshape(-1)
    if raw_labels.size == 0 or not np.isfinite(raw_labels).all():
        raise ValueError("labels must be finite and non-empty")
    labels = raw_labels.astype(np.int64)
    if scores.shape[0] != labels.shape[0] or scores.shape[0] == 0:
        raise ValueError("scores and labels must be non-empty arrays with equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores, labels


def _validate_logits(logits: np.ndarray, name: str) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, C] array")
    if not np.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite values")
    return logits


def _roc_metrics(y_true: np.ndarray, scores: np.ndarray, positive_label: int = 1) -> Dict[str, object]:
    scores, y_true = _validate_scores_labels(scores, y_true)
    if np.unique(y_true).size != 2:
        raise ValueError("ROC metrics require both positive and negative examples")
    fpr, tpr, thresholds = roc_curve(y_true == positive_label, scores)
    fnr = 1.0 - tpr
    delta = fpr - fnr
    crossing = np.flatnonzero(delta >= 0)
    if crossing.size and crossing[0] > 0:
        right = int(crossing[0])
        left = right - 1
        weight = float(-delta[left] / (delta[right] - delta[left])) if delta[right] != delta[left] else 0.0
        eer = float(fpr[left] + weight * (fpr[right] - fpr[left]))
        eer_threshold = float(thresholds[left] + weight * (thresholds[right] - thresholds[left]))
    else:
        eer_idx = int(np.argmin(np.abs(delta)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_threshold = float(thresholds[eer_idx])
    far_targets = (0.001, 0.01, 0.05, 0.10)
    tpr_at_far = {}
    for far in far_targets:
        valid = np.flatnonzero(fpr <= far)
        tpr_at_far[f"tpr_at_far_{far:g}"] = float(tpr[valid[-1]]) if valid.size else 0.0
    return {
        "auroc": float(auc(fpr, tpr)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tpr_at_far": tpr_at_far,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def evaluate_identification(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate closed-set device identification from classifier logits."""
    result = evaluate_logits(logits, labels)
    result["n_samples"] = float(np.asarray(labels).size)
    return result


def evaluate_model_identification(
    model: nn.Module, dataset, device: torch.device
) -> Dict[str, float]:
    """Collect logits from real IQ samples and evaluate closed-set identification."""
    logits, labels = collect_logits(model, dataset, device)
    return evaluate_identification(logits, labels)


def _cosine_scores(embeddings: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return np.sum(z[pairs[:, 0]] * z[pairs[:, 1]], axis=1)


def evaluate_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pairs: Optional[np.ndarray] = None,
    max_pairs: int = 100_000,
    seed: int = 0,
) -> Dict[str, object]:
    """Compute cosine-similarity verification ROC, EER, and TPR-at-FAR.

    ``pairs`` may be an ``[M, 2]`` index array. If omitted, a reproducible,
    approximately balanced set of genuine and impostor pairs is sampled.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    if z.ndim != 2 or z.shape[0] != y.shape[0] or z.shape[0] < 2 or z.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty [N, D] array aligned with labels")
    if not np.isfinite(z).all() or y.size == 0:
        raise ValueError("embeddings and labels must be finite and non-empty")
    if pairs is None:
        rng = np.random.default_rng(seed)
        if max_pairs < 2:
            raise ValueError("max_pairs must be at least 2")
        by_label = [np.flatnonzero(y == label) for label in np.unique(y)]
        genuine_classes = [indices for indices in by_label if indices.size >= 2]
        if len(by_label) < 2 or not genuine_classes:
            raise ValueError("verification requires at least one genuine and impostor pair")
        n_each = max_pairs // 2
        genuine_set = set()
        impostor_set = set()
        attempts = 0
        limit = max(1000, 30 * n_each)
        while len(genuine_set) < n_each and attempts < limit:
            indices = genuine_classes[int(rng.integers(len(genuine_classes)))]
            a, b = rng.choice(indices, 2, replace=False)
            genuine_set.add((min(int(a), int(b)), max(int(a), int(b))))
            attempts += 1
        attempts = 0
        while len(impostor_set) < n_each and attempts < limit:
            first, second = rng.choice(len(by_label), 2, replace=False)
            a = int(rng.choice(by_label[first]))
            b = int(rng.choice(by_label[second]))
            impostor_set.add((min(a, b), max(a, b)))
            attempts += 1
        if not genuine_set or not impostor_set:
            raise ValueError("verification requires at least one genuine and impostor pair")
        pairs = np.asarray(list(genuine_set) + list(impostor_set), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any((pairs < 0) | (pairs >= z.shape[0])):
        raise ValueError("pairs must have shape [M, 2] with valid embedding indices")
    pair_labels = (y[pairs[:, 0]] == y[pairs[:, 1]]).astype(np.int64)
    return _roc_metrics(pair_labels, _cosine_scores(z, pairs))


def evaluate_open_set(id_logits: np.ndarray, ood_logits: np.ndarray) -> Dict[str, object]:
    """Evaluate ID-vs-OOD detection using max-softmax confidence."""
    id_logits = _validate_logits(id_logits, "id_logits")
    ood_logits = _validate_logits(ood_logits, "ood_logits")
    if id_logits.shape[1] != ood_logits.shape[1]:
        raise ValueError("ID and OOD logits must have matching class dimensions")
    id_score = max_softmax_scores(id_logits)
    ood_score = max_softmax_scores(ood_logits)
    y = np.concatenate([np.ones(id_score.size, dtype=np.int64), np.zeros(ood_score.size, dtype=np.int64)])
    return _roc_metrics(y, np.concatenate([id_score, ood_score]))


def open_set_scores(logits: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
    """Return alternative confidence scores for open-set detection.

    Scores are oriented so larger values indicate greater known-device
    confidence. Energy returns ``-E_free = T*logsumexp(logits/T)``, which
    is higher for known devices (Liu et al. define ``E_free`` as lower
    for known).
    """
    logits = _validate_logits(logits, "logits")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    ordered = np.sort(logits, axis=1)
    neg_energy = temperature * (
        np.log(np.sum(np.exp(shifted), axis=1)) + np.max(scaled, axis=1)
    )
    return {
        "max_softmax": np.max(probabilities, axis=1),
        "logit_margin": ordered[:, -1] - ordered[:, -2] if logits.shape[1] >= 2 else ordered[:, -1],
        "negative_entropy": np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1),
        "energy": neg_energy,
    }


def prototype_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Return maximum cosine similarity to known-device prototypes."""
    reference = np.asarray(train_embeddings, dtype=np.float64)
    labels = np.asarray(train_labels).reshape(-1)
    query = np.asarray(embeddings, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("embedding arrays must be two-dimensional with matching widths")
    if labels.shape[0] != reference.shape[0] or reference.shape[0] == 0:
        raise ValueError("train_labels must align with non-empty train_embeddings")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("embedding arrays must be finite")
    classes = np.unique(labels)
    prototypes = np.stack([reference[labels == label].mean(axis=0) for label in classes])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    return np.max(query @ prototypes.T, axis=1)


def mahalanobis_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    embeddings: np.ndarray,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Return negative minimum Mahalanobis distance to known-class Gaussians.

    Fits class means and a shared covariance (shrunk toward its diagonal) on
    the training embeddings. Higher scores indicate greater known-device
    confidence, matching the orientation of :func:`prototype_scores`.
    """
    reference = np.asarray(train_embeddings, dtype=np.float64)
    labels = np.asarray(train_labels).reshape(-1)
    query = np.asarray(embeddings, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("embedding arrays must be two-dimensional with matching widths")
    if labels.shape[0] != reference.shape[0] or reference.shape[0] == 0:
        raise ValueError("train_labels must align with non-empty train_embeddings")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("embedding arrays must be finite")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    classes = np.unique(labels)
    means = np.stack([reference[labels == cls].mean(axis=0) for cls in classes])
    class_index = np.searchsorted(classes, labels)
    centered = reference - means[class_index]
    cov = centered.T @ centered / max(reference.shape[0] - 1, 1)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    cov += 1e-6 * np.eye(cov.shape[0])
    precision = np.linalg.pinv(cov)
    diff = query[:, None, :] - means[None, :, :]
    dist2 = np.einsum("ncd,de,nce->nc", diff, precision, diff)
    return -np.sqrt(np.maximum(dist2, 0.0)).min(axis=1)


def knn_scores(
    train_embeddings: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Return mean cosine similarity to the k nearest train embeddings.

    Unlike prototype cosine, kNN scoring respects local, potentially
    multi-modal cluster structure. Higher scores indicate greater
    known-device confidence.
    """
    reference = np.asarray(train_embeddings, dtype=np.float64)
    query = np.asarray(embeddings, dtype=np.float64)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("embedding arrays must be two-dimensional with matching widths")
    if reference.shape[0] == 0 or query.shape[0] == 0:
        raise ValueError("embedding arrays must be non-empty")
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("embedding arrays must be finite")
    if k < 1:
        raise ValueError("k must be at least 1")
    k = min(k, reference.shape[0])
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query @ reference.T
    top_k = np.partition(similarity, -k, axis=1)[:, -k:]
    return top_k.mean(axis=1)


def calibrate_rejection_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_unknown_far: float = 0.05,
) -> float:
    """Choose a threshold that maximizes known coverage at target unknown FAR."""
    known_scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    if known_scores.size == 0 or unknown_scores.size == 0:
        raise ValueError("calibration score arrays must be non-empty")
    if not 0.0 < target_unknown_far < 1.0:
        raise ValueError("target_unknown_far must be between 0 and 1")
    if not np.isfinite(known_scores).all() or not np.isfinite(unknown_scores).all():
        raise ValueError("calibration scores must be finite")

    candidates = np.unique(np.r_[known_scores, unknown_scores])
    unknown_far = np.array([np.mean(unknown_scores >= t) for t in candidates])
    known_coverage = np.array([np.mean(known_scores >= t) for t in candidates])

    feasible = np.flatnonzero(unknown_far <= target_unknown_far)
    if feasible.size == 0:
        best = int(np.argmin(unknown_far))
        return float(candidates[best])

    feasible_coverage = known_coverage[feasible]
    best = int(feasible[np.argmax(feasible_coverage)])
    return float(candidates[best])


def rejection_sweep(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    far_targets: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50),
) -> Dict[float, Dict[str, float]]:
    """Return rejection operating points at several FAR budgets."""
    known_scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    if known_scores.size == 0 or unknown_scores.size == 0:
        raise ValueError("score arrays must be non-empty")
    if not np.isfinite(known_scores).all() or not np.isfinite(unknown_scores).all():
        raise ValueError("scores must be finite")
    targets = [float(t) for t in far_targets]
    if any(not (0.0 < t < 1.0) for t in targets):
        raise ValueError("all far_targets must lie in the open interval (0, 1)")

    candidates = np.unique(np.r_[known_scores, unknown_scores])
    unknown_far = np.array([np.mean(unknown_scores >= t) for t in candidates])
    known_coverage = np.array([np.mean(known_scores >= t) for t in candidates])

    operating_points: Dict[float, Dict[str, float]] = {}
    for target in targets:
        feasible = np.flatnonzero(unknown_far <= target)
        if feasible.size == 0:
            idx = int(np.argmin(unknown_far))
        else:
            feasible_coverage = known_coverage[feasible]
            idx = int(feasible[np.argmax(feasible_coverage)])
        threshold = float(candidates[idx])
        operating_points[target] = {
            "threshold": threshold,
            "known_coverage": float(known_coverage[idx]),
            "unknown_far": float(unknown_far[idx]),
            "known_acceptance_rate": float(known_coverage[idx]),
        }
    return operating_points


def evaluate_rejection(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_labels: Optional[np.ndarray] = None,
    known_predictions: Optional[np.ndarray] = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Evaluate known coverage and unknown false acceptance at a threshold."""
    known_scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    known_accept = known_scores >= threshold
    unknown_accept = unknown_scores >= threshold
    result = {
        "threshold": float(threshold),
        "known_coverage": float(np.mean(known_accept)),
        "unknown_false_accept_rate": float(np.mean(unknown_accept)),
        "unknown_rejection_rate": float(1.0 - np.mean(unknown_accept)),
    }
    if known_labels is not None and known_predictions is not None:
        labels = np.asarray(known_labels).reshape(-1)
        predictions = np.asarray(known_predictions).reshape(-1)
        if labels.shape != predictions.shape or labels.shape[0] != known_scores.shape[0]:
            raise ValueError("known labels, predictions, and scores must align")
        result["accepted_known_accuracy"] = float(
            np.mean(predictions[known_accept] == labels[known_accept])
        ) if np.any(known_accept) else float("nan")
    return result


def evaluate_prototype_oscr(
    known_scores: np.ndarray,
    known_labels: np.ndarray,
    known_predictions: np.ndarray,
    unknown_scores: np.ndarray,
) -> Dict[str, object]:
    """Compute OSCR using prototype confidence scores and classifier labels."""
    known_scores = np.asarray(known_scores, dtype=np.float64).reshape(-1)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(known_labels).reshape(-1)
    predictions = np.asarray(known_predictions).reshape(-1)
    if known_scores.size == 0 or unknown_scores.size == 0:
        raise ValueError("known and unknown prototype scores must be non-empty")
    if labels.shape != predictions.shape or labels.shape[0] != known_scores.shape[0]:
        raise ValueError("known labels, predictions, and scores must align")
    if not np.isfinite(known_scores).all() or not np.isfinite(unknown_scores).all():
        raise ValueError("prototype scores must be finite")
    correct = predictions == labels
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[known_scores, unknown_scores]))[::-1], -np.inf]
    ccr = np.array([np.mean(correct & (known_scores >= threshold)) for threshold in thresholds])
    far = np.array([np.mean(unknown_scores >= threshold) for threshold in thresholds])
    order = np.argsort(far)
    return {
        "far": far[order],
        "ccr": ccr[order],
        "oscr_auc": float(auc(far[order], ccr[order])),
        "thresholds": thresholds[order],
    }


def compare_open_set_scores(
    known_scores: Dict[str, np.ndarray], unknown_scores: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, object]]:
    """Evaluate alternative known-confidence scores with common ROC metrics."""
    if set(known_scores) != set(unknown_scores):
        raise ValueError("known and unknown score dictionaries must have matching keys")
    return {
        name: _roc_metrics(
            np.concatenate([np.ones(len(known), dtype=np.int64), np.zeros(len(unknown), dtype=np.int64)]),
            np.concatenate([known, unknown]),
        )
        for name, (known, unknown) in ((key, (known_scores[key], unknown_scores[key])) for key in known_scores)
    }


def evaluate_oscr(
    id_logits: np.ndarray,
    id_labels: np.ndarray,
    ood_logits: np.ndarray,
) -> Dict[str, object]:
    """Compute the open-set classification rate (OSCR) curve and area."""
    id_logits = _validate_logits(id_logits, "id_logits")
    ood_logits = _validate_logits(ood_logits, "ood_logits")
    if id_logits.shape[1] != ood_logits.shape[1]:
        raise ValueError("ID and OOD logits must have matching class dimensions")
    id_labels = np.asarray(id_labels).reshape(-1)
    if id_labels.shape[0] != id_logits.shape[0]:
        raise ValueError("id_labels must align with ID logits")
    if id_labels.size == 0 or not np.isfinite(id_labels).all():
        raise ValueError("id_labels must be finite and non-empty")
    if not np.equal(id_labels, id_labels.astype(np.int64)).all() or np.any(
        (id_labels < 0) | (id_labels >= id_logits.shape[1])
    ):
        raise ValueError("id_labels must be integer class indices in [0, number of classes)")
    id_scores = max_softmax_scores(id_logits)
    ood_scores = max_softmax_scores(ood_logits)
    correct = np.argmax(id_logits, axis=1) == id_labels
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[id_scores, ood_scores]))[::-1], -np.inf]
    ccr = np.array([np.mean(correct & (id_scores >= t)) for t in thresholds])
    far = np.array([np.mean(ood_scores >= t) for t in thresholds])
    order = np.argsort(far)
    return {"oscr_auc": float(auc(far[order], ccr[order])), "far": far[order], "ccr": ccr[order], "thresholds": thresholds[order]}
