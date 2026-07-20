"""Configuration helpers for RF fingerprinting experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class DeviceImpairments:
    """Persistent transmitter impairments sampled once per device."""

    iq_imbalance: bool = True
    cfo: bool = True
    phase_noise: bool = True
    pa_nonlinearity: bool = True
    pa_memory: bool = True
    dc_offset: bool = False
    iq_gain_std: float = 0.06
    iq_skew_std_rad: float = 0.06
    cfo_range_rad: tuple[float, float] = (-0.012, 0.012)
    phase_noise_range: tuple[float, float] = (0.0008, 0.0025)
    pa_range: tuple[float, float] = (0.03, 0.11)
    memory_range: tuple[float, float] = (-0.06, 0.06)
    dc_std: float = 0.025


@dataclass
class NuisanceImpairments:
    """Capture and receiver impairments that should not identify a device."""

    session_channel: bool = True
    channel: bool = True
    receiver: bool = True
    waveform_variation: bool = True
    awgn: bool = True
    quantization: bool = True
    session_tap_scale: float = 0.10
    channel_tap_scale: float = 0.12
    snr_db_range: tuple[float, float] = (16.0, 27.0)


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
    impairment_profile: str = "default"
    impairment_ablation: Optional[str] = None
    device_impairments: DeviceImpairments = field(default_factory=DeviceImpairments)
    nuisance_impairments: NuisanceImpairments = field(default_factory=NuisanceImpairments)
    known_device_count: Optional[int] = None
    unknown_min_separation: float = 0.0

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
        self.validate_impairments()

    def validate_impairments(self) -> None:
        """Validate nested impairment settings after notebook overrides."""
        device = self.device_impairments
        nuisance = self.nuisance_impairments
        if self.known_device_count is not None and not 1 <= self.known_device_count < self.n_devices:
            raise ValueError("known_device_count must be in [1, n_devices).")
        if self.unknown_min_separation < 0:
            raise ValueError("unknown_min_separation must be non-negative.")
        for name, value in (
            ("iq_gain_std", device.iq_gain_std),
            ("iq_skew_std_rad", device.iq_skew_std_rad),
            ("phase_noise_min", device.phase_noise_range[0]),
            ("phase_noise_max", device.phase_noise_range[1]),
            ("pa_min", device.pa_range[0]),
            ("pa_max", device.pa_range[1]),
            ("dc_std", device.dc_std),
            ("session_tap_scale", nuisance.session_tap_scale),
            ("channel_tap_scale", nuisance.channel_tap_scale),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name, bounds in (
            ("cfo_range_rad", device.cfo_range_rad),
            ("phase_noise_range", device.phase_noise_range),
            ("pa_range", device.pa_range),
            ("memory_range", device.memory_range),
            ("snr_db_range", nuisance.snr_db_range),
        ):
            if len(bounds) != 2 or not all(isinstance(v, (int, float)) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be a finite (min, max) range.")


def _impairment_profile(name: str) -> tuple[DeviceImpairments, NuisanceImpairments]:
    """Return independent impairment settings for a named experiment profile."""
    device = DeviceImpairments()
    nuisance = NuisanceImpairments()
    if name == "oracle":
        return replace(device, iq_imbalance=False, cfo=False, phase_noise=False,
                       pa_nonlinearity=False, pa_memory=False, dc_offset=False), replace(
                           nuisance, session_channel=False, channel=False,
                           receiver=False, waveform_variation=False, awgn=False,
                           quantization=False)
    if name == "device_full":
        return replace(device, dc_offset=False), replace(
            nuisance, session_channel=False, channel=False, receiver=False,
            waveform_variation=False, awgn=False, quantization=False)
    if name == "controlled":
        return replace(device, cfo_range_rad=(-0.003, 0.003), iq_gain_std=0.03,
                       iq_skew_std_rad=0.03, phase_noise_range=(0.0003, 0.0012),
                       pa_range=(0.02, 0.08), memory_range=(-0.04, 0.04),
                       dc_offset=False), replace(
                           nuisance, session_channel=False, channel=False,
                           waveform_variation=False)
    if name == "receiver_only":
        return replace(device, cfo_range_rad=(-0.003, 0.003), iq_gain_std=0.03,
                       iq_skew_std_rad=0.03, phase_noise_range=(0.0003, 0.0012),
                       pa_range=(0.02, 0.08), memory_range=(-0.04, 0.04),
                       dc_offset=False), replace(
                           nuisance, session_channel=False, channel=False,
                           waveform_variation=False, receiver=True, awgn=True,
                           quantization=True)
    if name == "stress_channel":
        return replace(device, cfo_range_rad=(-0.003, 0.003), iq_gain_std=0.03,
                       iq_skew_std_rad=0.03, phase_noise_range=(0.0003, 0.0012),
                       pa_range=(0.02, 0.08), memory_range=(-0.04, 0.04),
                       dc_offset=False), replace(
                           nuisance, session_channel=True, channel=False,
                           waveform_variation=False)
    if name == "stress_waveform":
        return replace(device, cfo_range_rad=(-0.003, 0.003), iq_gain_std=0.03,
                       iq_skew_std_rad=0.03, phase_noise_range=(0.0003, 0.0012),
                       pa_range=(0.02, 0.08), memory_range=(-0.04, 0.04),
                       dc_offset=False), replace(
                           nuisance, session_channel=False, channel=False,
                           waveform_variation=True)
    if name == "full":
        return replace(device, dc_offset=True), nuisance
    if name == "default":
        # Preserve the pre-profile generator behavior for existing callers.
        return replace(device, dc_offset=True), nuisance
    raise ValueError("Unknown impairment profile. Choose oracle, device_full, controlled, receiver_only, stress_channel, stress_waveform, default, or full.")


def _apply_impairment_ablation(cfg: RFConfig) -> None:
    """Disable one named source impairment without changing other settings."""
    name = cfg.impairment_ablation
    if not name:
        return
    device = cfg.device_impairments
    nuisance = cfg.nuisance_impairments
    mapping = {
        "no_iq": ("device", "iq_imbalance"),
        "no_cfo": ("device", "cfo"),
        "no_phase_noise": ("device", "phase_noise"),
        "no_pa": ("device", "pa_nonlinearity"),
        "no_memory": ("device", "pa_memory"),
        "no_dc": ("device", "dc_offset"),
        "no_session_channel": ("nuisance", "session_channel"),
        "no_channel": ("nuisance", "channel"),
        "no_receiver": ("nuisance", "receiver"),
        "no_waveform_variation": ("nuisance", "waveform_variation"),
        "no_awgn": ("nuisance", "awgn"),
        "no_quantization": ("nuisance", "quantization"),
    }
    if name not in mapping:
        raise ValueError(f"Unknown impairment ablation: {name}")
    group, field_name = mapping[name]
    if group == "device":
        cfg.device_impairments = replace(device, **{field_name: False})
    else:
        cfg.nuisance_impairments = replace(nuisance, **{field_name: False})

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


def load_config(mode: str = "base", impairment_profile: str = "default",
                impairment_ablation: Optional[str] = None) -> RFConfig:
    """Load a configuration preset.

    Args:
        mode: Runtime preset (``fast``, ``base``, or ``full``).

    Returns:
        Populated :class:`RFConfig`.
    """
    device_impairments, nuisance_impairments = _impairment_profile(impairment_profile)
    cfg = RFConfig(
        impairment_profile=impairment_profile,
        impairment_ablation=impairment_ablation,
        device_impairments=device_impairments,
        nuisance_impairments=nuisance_impairments,
    )
    for key, value in _preset_overrides(mode).items():
        setattr(cfg, key, value)

    _apply_impairment_ablation(cfg)
    cfg.validate_impairments()

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
