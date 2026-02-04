"""Training script for multimodal model with projection layers."""
import os
# Disable tkinter backend for matplotlib/PIL to prevent multiprocessing crashes
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

from config import Config
from multimodal_model import MultiModalModel
from metrics import MetricsTracker
from dataset import CocoCaptionsDataset


def collate_fn(batch, tokenizer):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]

    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding_len = max_len - seq_len

        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
        ])
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(padding_len, dtype=torch.long)
        ])
        padded_labels = torch.cat([
            item["labels"],
            torch.full((padding_len,), -100, dtype=torch.long)
        ])

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
        labels_list.append(padded_labels)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "captions": captions,
    }


def setup_dataloaders(config: Config, tokenizer, image_processor) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CocoCaptionsDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="train",
        max_length=config.max_length,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        prefetch_size=config.prefetch_size,
        prefetch_workers=config.num_workers,
        max_samples=config.train_max_samples,
    )
    val_dataset = CocoCaptionsDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        split="validation",
        max_length=config.max_length,
        system_prompt=config.system_prompt,
        user_prompt=config.user_prompt,
        prefetch_size=config.prefetch_size,
        prefetch_workers=config.num_workers,
        max_samples=config.val_max_samples if config.val_max_samples > 0 else -1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, scaler, config: Config, epoch: int) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        optimizer.zero_grad()

        # Always use autocast for mixed precision, scaler is only for loss scaling
        with autocast(device_type="cuda", enabled=config.mixed_precision):
            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs.loss
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if step % config.log_interval == 0 and step > 0:
            avg_loss = total_loss / num_batches
            logging.info(f"Epoch {epoch} Step {step}/{len(dataloader)} | Loss: {avg_loss:.4f}")

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate(model, dataloader, config: Config, epoch: int) -> Dict[str, float]:
    model.eval()
    tracker = MetricsTracker()
    total_loss = 0.0
    num_batches = 0

    max_batches = config.val_max_samples // config.batch_size if config.val_max_samples > 0 else len(dataloader)
    max_batches = min(max_batches, len(dataloader))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", total=max_batches)
    for step, batch in enumerate(pbar):
        if step >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)
        captions = batch["captions"]

        with autocast(device_type="cuda", enabled=config.mixed_precision):
            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs.loss

        total_loss += loss.item()
        num_batches += 1

        try:
            generated = model.generate_from_pixel_values(
                pixel_values,
                config.device,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                system_prompt=config.system_prompt,
                user_prompt=config.user_prompt,
            )
            references = [[cap] for cap in captions]
            tracker.update(generated, references, loss.item())
        except Exception as exc:
            logging.warning(f"Generation failed: {exc}")
            tracker.update(["" for _ in captions], [[cap] for cap in captions], loss.item())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = tracker.compute()
    metrics["loss"] = total_loss / num_batches
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, output_dir: Path, is_best: bool = False):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(state, epoch_path)
    logging.info(f"Saved checkpoint to {epoch_path}")

    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(state, best_path)
        logging.info(f"Saved best model to {best_path}")


def plot_metrics(metrics_history: List[dict], output_dir: Path):
    if len(metrics_history) < 2:
        return

    epochs = [m["epoch"] for m in metrics_history]
    train_losses = [m.get("train_loss", 0) for m in metrics_history]
    val_losses = [m.get("val_loss", 0) for m in metrics_history]
    ciders = [m.get("cider", 0) for m in metrics_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, ciders, label="CIDEr", marker="o", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CIDEr Score")
    ax2.set_title("CIDEr Score Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal model with projection layers")

    # Checkpoint
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Model settings
    parser.add_argument(
        "--projection_type",
        type=str,
        choices=["mlp", "qformer", "perceiver"],
        help="Type of projection layer to use",
    )
    parser.add_argument("--freeze_vision", type=lambda x: x.lower() == "true", help="Freeze vision model weights")
    parser.add_argument("--freeze_llm", type=lambda x: x.lower() == "true", help="Freeze LLM weights")

    # Training settings
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, help="Gradient clipping value")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")

    # Data settings
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, help="DataLoader workers for parallel data loading")
    parser.add_argument("--prefetch_size", type=int, help="Number of images to prefetch ahead")
    parser.add_argument("--prefetch_workers", type=int, help="Concurrent async download tasks per worker")
    parser.add_argument("--system_prompt", type=str, help="System prompt for the model")
    parser.add_argument("--user_prompt", type=str, help="User prompt for image description")
    parser.add_argument("--train_max_samples", type=int, help="Max training samples (-1 for all)")
    parser.add_argument("--val_max_samples", type=int, help="Max validation samples (-1 for all)")

    # Validation settings
    parser.add_argument("--val_interval", type=int, help="Validate every N epochs")

    # System settings
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--mixed_precision", type=lambda x: x.lower() == "true", help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Logging settings
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints and logs")
    parser.add_argument("--log_interval", type=int, help="Steps between logging")
    parser.add_argument("--save_interval", type=int, help="Epochs between checkpoints")
    parser.add_argument("--keep_last_n", type=int, help="Keep last N checkpoints")

    # Inference settings
    parser.add_argument("--max_new_tokens", type=int, help="Max tokens to generate during validation")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Nucleus sampling top-p value")
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", help="Enable sampling during generation")

    args = parser.parse_args()

    # Create config with defaults
    config = Config()

    # Override config with CLI arguments
    for key, value in vars(args).items():
        if key != "resume" and value is not None:
            setattr(config, key, value)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Creating model...")
    model = MultiModalModel(config, projection_type=config.projection_type).to(device)

    if config.freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
    if config.freeze_llm:
        for param in model.llm.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    logging.info("Setting up dataloaders...")
    train_loader, val_loader = setup_dataloaders(
        config, model.tokenizer, model.image_processor
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    # GradScaler doesn't work with BFloat16, only with FP16
    # Check LLM specifically since that's what matters for GradScaler
    is_bfloat16 = next(model.llm.parameters()).dtype == torch.bfloat16
    if config.mixed_precision and not is_bfloat16:
        scaler = GradScaler()
        logging.info("Using GradScaler for FP16 mixed precision")
    else:
        scaler = None
        if is_bfloat16:
            logging.info("BFloat16 detected, GradScaler disabled (not needed)")
        elif not config.mixed_precision:
            logging.info("Mixed precision disabled")

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        logging.info(f"Resumed from {args.resume} (epoch {state['epoch']})")

    best_cider = 0.0
    metrics_history = []

    for epoch in range(start_epoch, config.num_epochs + 1):
        logging.info("\n" + "=" * 50)
        logging.info(f"Epoch {epoch}/{config.num_epochs}")
        logging.info("=" * 50)

        train_metrics = train_epoch(model, train_loader, optimizer, scaler, config, epoch)
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}")

        if epoch % config.val_interval == 0:
            val_metrics = validate(model, val_loader, config, epoch)
            logging.info(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"CIDEr: {val_metrics['cider']:.4f} | "
                f"METEOR: {val_metrics['meteor']:.4f}"
            )

            metrics_history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "cider": val_metrics["cider"],
                "meteor": val_metrics["meteor"],
            })

            is_best = val_metrics["cider"] > best_cider
            if is_best:
                best_cider = val_metrics["cider"]

            if epoch % config.save_interval == 0 or is_best:
                save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, is_best)

            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_history, f, indent=2)

            plot_metrics(metrics_history, output_dir)

    logging.info("\nTraining complete!")
    logging.info(f"Best CIDEr: {best_cider:.4f}")


if __name__ == "__main__":
    main()
