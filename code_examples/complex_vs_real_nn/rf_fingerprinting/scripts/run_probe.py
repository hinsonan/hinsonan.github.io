"""Run linear probe for RF fingerprinting encoders."""

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
from rf_fingerprinting.models_complex import ComplexEncoder1D
from rf_fingerprinting.models_real import RealEncoder1D
from rf_fingerprinting.probe import run_linear_probe


def main():
    """Run linear probe from CLI arguments.

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
    test_ds = IQDataset(data["iq"][splits["test"]], data["device_id"][splits["test"]])

    encoder = RealEncoder1D(cfg.embed_dim) if args.encoder == "real" else ComplexEncoder1D(cfg.embed_dim)
    ckpt = Path(cfg.output_dir) / f"{args.encoder}_encoder_pretrained.pt"
    if ckpt.exists():
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
    else:
        print(f"Warning: missing pretrained checkpoint at {ckpt}. Running probe with random encoder.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    metrics = run_linear_probe(encoder, train_ds, test_ds, device)
    print(metrics)


if __name__ == "__main__":
    main()
