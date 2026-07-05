"""Run SimCLR pretraining for RF fingerprinting encoders."""

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
from rf_fingerprinting.datasets import TwoViewIQDataset, split_indices
from rf_fingerprinting.logging_utils import save_json
from rf_fingerprinting.models_complex import ComplexEncoder1D
from rf_fingerprinting.models_real import RealEncoder1D
from rf_fingerprinting.paths import ensure_dir
from rf_fingerprinting.pretrain import pretrain_simclr


def main():
    """Run pretraining from CLI arguments.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rf_fp_fast.yaml")
    parser.add_argument("--encoder", choices=["real", "complex"], default="real")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_or_generate_npz(cfg)
    splits = split_indices(data["iq"].shape[0], cfg.test_size, cfg.val_size, cfg.seed)
    train_iq = data["iq"][splits["train"]]

    ds = TwoViewIQDataset(train_iq, cfg)
    encoder = RealEncoder1D(cfg.embed_dim) if args.encoder == "real" else ComplexEncoder1D(cfg.embed_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist = pretrain_simclr(encoder, ds, cfg, device)

    out_dir = ensure_dir(cfg.output_dir)
    torch.save(encoder.state_dict(), out_dir / f"{args.encoder}_encoder_pretrained.pt")
    save_json(str(out_dir / f"pretrain_{args.encoder}_history.json"), hist)
    print({"encoder": args.encoder, "final_pretrain_loss": hist["loss"][-1]})


if __name__ == "__main__":
    main()
