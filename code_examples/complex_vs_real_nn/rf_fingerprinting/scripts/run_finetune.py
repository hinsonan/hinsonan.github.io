"""Run supervised fine-tuning for RF fingerprinting encoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rf_fingerprinting.config import load_config
from rf_fingerprinting.data_io import load_or_generate_npz
from rf_fingerprinting.datasets import IQDataset, split_indices
from rf_fingerprinting.finetune import finetune_classifier
from rf_fingerprinting.logging_utils import save_json
from rf_fingerprinting.models_complex import ComplexEncoder1D
from rf_fingerprinting.models_real import RealEncoder1D
from rf_fingerprinting.paths import ensure_dir


def main():
    """Run fine-tuning from CLI arguments.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rf_fp_fast.yaml")
    parser.add_argument("--encoder", choices=["real", "complex"], default="real")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_or_generate_npz(cfg)
    splits = split_indices(
        data["iq"].shape[0],
        cfg.test_size,
        cfg.val_size,
        cfg.seed,
        labels=data["device_id"],
    )

    train_ds = IQDataset(data["iq"][splits["train"]], data["device_id"][splits["train"]])
    val_ds = IQDataset(data["iq"][splits["val"]], data["device_id"][splits["val"]])
    num_classes = int(data["device_id"].max()) + 1

    encoder = RealEncoder1D(cfg.embed_dim) if args.encoder == "real" else ComplexEncoder1D(cfg.embed_dim)
    ckpt = Path(cfg.output_dir) / f"{args.encoder}_encoder_pretrained.pt"
    if ckpt.exists():
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"Warning: missing pretrained checkpoint at {ckpt}. Fine-tuning from random initialization.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hist = finetune_classifier(encoder.to(device), train_ds, val_ds, num_classes, cfg, device)

    out_dir = ensure_dir(cfg.output_dir)
    torch.save(model.state_dict(), out_dir / f"{args.encoder}_finetuned.pt")
    save_json(str(out_dir / f"finetune_{args.encoder}_history.json"), hist)
    print({"encoder": args.encoder, "best_val_acc": max(hist["val_acc"] or [0.0])})


if __name__ == "__main__":
    main()
