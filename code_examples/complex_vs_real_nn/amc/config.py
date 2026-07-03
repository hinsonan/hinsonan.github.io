"""Configuration for the modulation-classification experiment."""

from dataclasses import dataclass


@dataclass
class ModClassConfig:
    """Configuration for rotation-invariant modulation classification."""

    burst_len: int = 128
    modulations: tuple = ("bpsk", "qpsk", "8psk", "16qam")

    train_phase_deg: float = 15.0
    full_phase_deg: float = 180.0

    snr_db: float = 10.0

    n_train: int = 12000
    n_val: int = 4000
    seed: int = 7

    complex_channels: tuple = (24, 48, 48)
    real_channels: tuple = (32, 64, 64)
    kernel_size: int = 7
    stride: int = 2
    hidden_dim: int = 128
    moment_hidden_dim: int = 48
    moment_orders: tuple = (2, 4, 8)

    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    out_dir: str = "trained_modclass"

    @property
    def n_classes(self) -> int:
        return len(self.modulations)
