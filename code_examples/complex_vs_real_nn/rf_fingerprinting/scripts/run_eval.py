"""Run evaluation for fine-tuned RF fingerprinting models."""

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
from rf_fingerprinting.evaluate import collect_logits, evaluate_logits
from rf_fingerprinting.finetune import EncoderClassifier
from rf_fingerprinting.models_complex import ComplexEncoder1D
from rf_fingerprinting.models_real import RealEncoder1D
from rf_fingerprinting.open_set import open_set_auc


def main():
    """Run evaluation from CLI arguments.

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
    test_ds = IQDataset(data["iq"][splits["test"]], data["device_id"][splits["test"]])
    train_ds = IQDataset(data["iq"][splits["train"]], data["device_id"][splits["train"]])

    num_classes = int(data["device_id"].max()) + 1
    encoder = RealEncoder1D(cfg.embed_dim) if args.encoder == "real" else ComplexEncoder1D(cfg.embed_dim)
    model = EncoderClassifier(encoder, cfg.embed_dim, num_classes)

    out_dir = Path(cfg.output_dir)
    ckpt = out_dir / f"{args.encoder}_finetuned.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_logits, test_labels = collect_logits(model, test_ds, device)
    id_metrics = evaluate_logits(test_logits, test_labels)

    train_logits, _ = collect_logits(model, train_ds, device)
    ood_logits = train_logits + 0.8 * torch.randn(*train_logits.shape).numpy()
    os_metrics = open_set_auc(test_logits, ood_logits)

    out = {}
    out.update(id_metrics)
    out.update(os_metrics)
    print(out)


if __name__ == "__main__":
    main()
