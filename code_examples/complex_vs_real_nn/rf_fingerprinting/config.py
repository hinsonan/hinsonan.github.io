"""Configuration helpers for RF fingerprinting experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class RFConfig:
    """Experiment configuration.

    The ``mode`` field selects a preset (``fast``, ``base``, or ``full``).
    Presets are applied inside :func:`load_config` and can be overridden
    by setting any field explicitly after loading.
    """

    mode: str = "base"
    dataset_path: str = ""
    output_dir: str = "outputs/rf_fingerprinting"
    seed: int = 7

    seq_len: int = 256
    n_devices: int = 6
    n_sessions: int = 3
    n_samples: int = 1200

    test_size: float = 0.2
    val_size: float = 0.2

    batch_size: int = 64
    embed_dim: int = 64
    pretrain_epochs: int = 4
    finetune_epochs: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.2

    noise_std: float = 0.04
    phase_jitter_rad: float = 0.05
    time_shift: int = 8
    cfo_jitter_rad: float = 0.0
    amplitude_jitter: float = 0.05
    aug_prob: float = 0.5

    grad_clip: float = 0.0
    dropout: float = 0.0
    encoder_lr_scale: float = 1.0
    warmup_epochs: int = 1

    # Appended fields preserve the positional order of the original config.
    n_receivers: int = 2
    num_workers: int = 0
    n_channels: int = 3
    normalize_output: bool = True
    quantization_scale: float = 2.0
    quantization_bits: int = 10

    def __post_init__(self) -> None:
        """Reject dimensions and probabilities that cannot produce a dataset."""
        for name in ("seq_len", "n_devices", "n_sessions", "n_receivers", "n_channels", "batch_size", "embed_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.n_samples <= 0 or self.num_workers < 0:
            raise ValueError("n_samples must be positive and num_workers non-negative.")
        if not 0.0 <= self.test_size <= 1.0 or not 0.0 <= self.val_size <= 1.0:
            raise ValueError("test_size and val_size must be between 0 and 1.")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.")
        if not 0.0 <= self.aug_prob <= 1.0:
            raise ValueError("aug_prob must be between 0 and 1.")
        if self.quantization_scale <= 0 or self.quantization_bits < 2:
            raise ValueError("quantization_scale must be positive and quantization_bits at least 2.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a plain dictionary."""
        return asdict(self)


def _preset_overrides(mode: str) -> Dict[str, Any]:
    """Return preset overrides for a named runtime mode.

    Args:
        mode: One of ``fast``, ``base``, or ``full``.

    Returns:
        Dictionary of field overrides.
    """
    if mode == "fast":
        return {
            "mode": "fast",
            "output_dir": "outputs/rf_fingerprinting_fast",
            "n_samples": 800,
            "n_sessions": 3,
            "n_receivers": 2,
            "pretrain_epochs": 2,
            "finetune_epochs": 5,
            "batch_size": 64,
            "noise_std": 0.05,
            "phase_jitter_rad": 0.05,
            "cfo_jitter_rad": 0.0,
            "amplitude_jitter": 0.05,
            "aug_prob": 0.5,
            "grad_clip": 0.0,
            "dropout": 0.0,
            "encoder_lr_scale": 1.0,
        }
    if mode == "full":
        return {
            "mode": "full",
            "output_dir": "outputs/rf_fingerprinting_full",
            "n_samples": 4000,
            "n_sessions": 4,
            "n_receivers": 2,
            "pretrain_epochs": 10,
            "finetune_epochs": 10,
            "batch_size": 128,
            "noise_std": 0.03,
            "phase_jitter_rad": 0.03,
            "cfo_jitter_rad": 0.0,
            "amplitude_jitter": 0.03,
            "aug_prob": 0.9,
            "grad_clip": 0.5,
            "dropout": 0.1,
            "encoder_lr_scale": 0.5,
        }
    return {
        "mode": "base",
        "output_dir": "outputs/rf_fingerprinting",
        "n_samples": 1200,
        "n_sessions": 3,
        "n_receivers": 2,
        "pretrain_epochs": 4,
        "finetune_epochs": 6,
        "batch_size": 64,
        "noise_std": 0.04,
        "phase_jitter_rad": 0.05,
        "cfo_jitter_rad": 0.0,
        "amplitude_jitter": 0.05,
        "aug_prob": 0.6,
        "grad_clip": 0.5,
        "dropout": 0.05,
        "encoder_lr_scale": 0.5,
    }


def load_config(mode: str = "base") -> RFConfig:
    """Load a configuration preset.

    Args:
        mode: Runtime preset (``fast``, ``base``, or ``full``).

    Returns:
        Populated :class:`RFConfig`.
    """
    cfg = RFConfig()
    for key, value in _preset_overrides(mode).items():
        setattr(cfg, key, value)

    if cfg.dataset_path and not Path(cfg.dataset_path).is_absolute():
        cfg.dataset_path = str(project_root() / cfg.dataset_path)
    if cfg.output_dir and not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str(project_root() / cfg.output_dir)
    return cfg


def project_root() -> Path:
    """Return the project root directory.

    Returns:
        Absolute path to ``.../rf_fingerprinting``.
    """
    return Path(__file__).resolve().parent


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it.

    Args:
        path: Directory path.

    Returns:
        Resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str, payload: Dict):
    """Save a dictionary to a JSON file.

    Args:
        path: Output JSON file path.
        payload: JSON-serializable dictionary.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_scorecard_csv(path: str, rows: Iterable[Dict[str, object]]):
    """Save experiment scorecard rows to CSV.

    Args:
        path: Output CSV path.
        rows: Iterable of dictionaries with identical keys.
    """
    rows = list(rows)
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
