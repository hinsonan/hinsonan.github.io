"""Evaluation logic for the AMC experiment."""

import os

import numpy as np
import torch

try:
    from .data import generate_dataset
    from .training import RUNS
except ImportError:
    from data import generate_dataset
    from training import RUNS


def load_run(run_name, cfg, model_dir, device):
    """Load a trained model checkpoint from disk.

    Args:
        run_name: Name of the run (sub-directory under ``model_dir``).
        cfg: Experiment configuration (determines model architecture).
        model_dir: Root directory containing run sub-directories.
        device: Device to load the model onto.

    Returns:
        The loaded model in eval mode, or ``None`` if the checkpoint
        could not be found or loaded.
    """
    try:
        from .models import build_model
    except ImportError:
        from models import build_model

    model_name = RUNS[run_name][0]
    ckpt = os.path.join(model_dir, run_name, "best_model.pt")
    if not os.path.exists(ckpt):
        return None
    model = build_model(model_name, cfg).to(device)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except Exception as exc:
        print(f"  skipped {run_name}: could not load checkpoint ({exc})")
        return None
    model.eval()
    return model


@torch.no_grad()
def accuracy_at_angle(model, cfg, angle_deg, snr_db, device, n=2000, per_class=False):
    """Evaluate top-1 accuracy at a fixed rotation angle.

    Args:
        model: Trained classifier.
        cfg: Experiment configuration.
        angle_deg: Test rotation angle in degrees (both low and high bound).
        snr_db: SNR in decibels.
        device: Device for inference.
        n: Number of test samples.
        per_class: If ``True``, return per-class accuracy list instead.

    Returns:
        Overall accuracy (float) or, if ``per_class=True``, a list of
        per-class accuracies.
    """
    data = generate_dataset(n, cfg, angle_deg, angle_deg, snr_db=snr_db, seed_offset=900)
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    pred = model(iq).argmax(1)
    if per_class:
        n_cls = cfg.n_classes
        accs = []
        for c in range(n_cls):
            m = label == c
            accs.append((pred[m] == label[m]).float().mean().item() if m.any() else float("nan"))
        return accs
    return (pred == label).float().mean().item()


@torch.no_grad()
def confusion_at_angle(model, cfg, angle_deg, snr_db, device, n=4000):
    """Compute the confusion matrix at a fixed rotation angle.

    Args:
        model: Trained classifier.
        cfg: Experiment configuration.
        angle_deg: Test rotation angle in degrees.
        snr_db: SNR in decibels.
        device: Device for inference.
        n: Number of test samples.

    Returns:
        Confusion matrix as an ``(n_classes, n_classes)`` int64 array.
    """
    data = generate_dataset(n, cfg, angle_deg, angle_deg, snr_db=snr_db, seed_offset=901)
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    pred = model(iq).argmax(1)
    k = cfg.n_classes
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(label.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm


@torch.no_grad()
def accuracy_full_circle(model, cfg, snr_db, device, n=4000):
    """Evaluate top-1 accuracy over the full rotation range.

    Samples are drawn uniformly from ``±full_phase_deg``.

    Args:
        model: Trained classifier.
        cfg: Experiment configuration.
        snr_db: SNR in decibels.
        device: Device for inference.
        n: Number of test samples.

    Returns:
        Overall accuracy as a float in ``[0, 1]``.
    """
    data = generate_dataset(
        n, cfg, -cfg.full_phase_deg, cfg.full_phase_deg, snr_db=snr_db, seed_offset=902
    )
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    return (model(iq).argmax(1) == label).float().mean().item()
