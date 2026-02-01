"""Configuration for multimodal model training and inference."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Training and inference configuration."""
    
    # Model settings
    projection_type: str = "mlp"  # mlp | qformer | perceiver
    freeze_vision: bool = True
    freeze_llm: bool = False
    
    # Training settings
    batch_size: int = 8
    num_epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    
    # Data settings
    max_length: int = 512
    num_workers: int = 4
    prefetch_size: int = 16
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    user_prompt: str = "Describe this image."
    train_max_samples: int = 64  # -1 = use all, >0 = limit train set for testing
    val_max_samples: int = 64   # Max validation samples (-1 for all)
    
    # Validation settings
    val_interval: int = 1  # Validate every N epochs
    
    # System settings
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
    
    # Logging settings
    output_dir: str = "outputs"
    log_interval: int = 50  # Steps between logging
    save_interval: int = 2  # Epochs between checkpoints
    keep_last_n: int = 3  # Keep last N checkpoints + best
    
    # Inference settings
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
