"""Configuration helpers for RF fingerprinting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import project_root


@dataclass
class RFConfig:
    """Experiment configuration.

    Attributes:
        mode: Preset name: ``fast``, ``base``, or ``full``.
        dataset_path: Optional path to an NPZ file with RF captures.
        output_dir: Directory for outputs and checkpoints.
        seed: Random seed.
        seq_len: Number of IQ samples per capture.
        n_devices: Number of synthetic devices if fallback data is used.
        n_sessions: Number of synthetic sessions.
        n_samples: Total synthetic samples.
        test_size: Fraction for test split.
        val_size: Fraction for validation split (from train partition).
        batch_size: DataLoader batch size.
        embed_dim: Embedding dimension for both encoders.
        pretrain_epochs: Number of SimCLR epochs.
        finetune_epochs: Number of supervised fine-tune epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        temperature: NT-Xent temperature.
        noise_std: Default augmentation noise standard deviation.
        phase_jitter_rad: Augmentation phase jitter range in radians.
        time_shift: Max absolute circular time shift.
    """

    mode: str = "base"
    dataset_path: str = ""
    output_dir: str = "outputs/rf_fingerprinting"
    seed: int = 7

    seq_len: int = 256
    n_devices: int = 6
    n_sessions: int = 2
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
    phase_jitter_rad: float = 0.35
    time_shift: int = 8
    cfo_jitter_rad: float = 0.02
    amplitude_jitter: float = 0.1
    aug_prob: float = 0.8

    grad_clip: float = 0.0
    dropout: float = 0.0
    encoder_lr_scale: float = 1.0
    warmup_epochs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


def _preset_overrides(mode: str) -> Dict[str, Any]:
    """Return config overrides for a named runtime preset.

    Args:
        mode: Preset name.

    Returns:
        A dictionary of overrides.
    """
    if mode == "fast":
        return {
            "mode": "fast",
            "n_samples": 800,
            "batch_size": 64,
            "pretrain_epochs": 2,
            "finetune_epochs": 5,
        }
    if mode == "full":
        return {
            "mode": "full",
            "n_samples": 4000,
            "batch_size": 128,
            "pretrain_epochs": 10,
            "finetune_epochs": 8,
        }
    return {"mode": "base"}


def load_config(path: Optional[str] = None, mode: str = "base") -> RFConfig:
    """Load configuration from preset and optional YAML/JSON file.

    Args:
        path: Optional config path. Relative paths are resolved against the
            rf_fingerprinting project root. YAML requires ``pyyaml``.
        mode: Runtime preset (``fast``, ``base``, ``full``).

    Returns:
        Populated ``RFConfig``.
    """
    cfg = RFConfig()
    for key, value in _preset_overrides(mode).items():
        setattr(cfg, key, value)

    if not path:
        return cfg

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    if cfg_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files.") from exc
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        import json

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if cfg.dataset_path and not Path(cfg.dataset_path).is_absolute():
        cfg.dataset_path = str(project_root() / cfg.dataset_path)
    if cfg.output_dir and not Path(cfg.output_dir).is_absolute():
        cfg.output_dir = str(project_root() / cfg.output_dir)
    return cfg
