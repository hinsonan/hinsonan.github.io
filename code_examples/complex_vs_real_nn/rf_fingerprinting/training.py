"""Training, probing, and evaluation routines for RF fingerprinting."""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

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
    from .models import EncoderClassifier, ProjectionHead
except ImportError:  # pragma: no cover - exercised by notebook-style imports.
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
