"""Training logic for the AMC experiment."""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import ModClassConfig
from data import generate_dataset
from models import build_model, count_parameters

BASE_DIR = Path(__file__).resolve().parent

RUNS = {
    "complex_narrow": ("complex", "tab:blue", "Complex (trained +/-15 deg)"),
    "complex_moment": ("complex_moment", "tab:purple", "Complex+moments (trained +/-15 deg)"),
    "real_narrow": ("real", "tab:red", "Real (trained +/-15 deg)"),
    "real_full": ("real", "tab:green", "Real (trained full circle)"),
}


def run_specs(cfg: ModClassConfig):
    return {
        "complex_narrow": ("complex", cfg.train_phase_deg),
        "complex_moment": ("complex_moment", cfg.train_phase_deg),
        "real_narrow": ("real", cfg.train_phase_deg),
        "real_full": ("real", cfg.full_phase_deg),
    }


def resolve_dir(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(BASE_DIR / path)


def _tensor_loader(data, batch_size, device, shuffle):
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    theta = torch.from_numpy(data["theta"]).to(device)
    ds = TensorDataset(iq, label, theta)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def make_loaders(cfg: ModClassConfig, train_phase_deg: float, device):
    train = generate_dataset(
        cfg.n_train, cfg, -train_phase_deg, train_phase_deg, seed_offset=0
    )
    val_ind = generate_dataset(
        cfg.n_val, cfg, -train_phase_deg, train_phase_deg, seed_offset=100
    )
    val_ood = generate_dataset(
        cfg.n_val, cfg, -cfg.full_phase_deg, cfg.full_phase_deg, seed_offset=200
    )
    return (
        _tensor_loader(train, cfg.batch_size, device, True),
        _tensor_loader(val_ind, cfg.batch_size, device, False),
        _tensor_loader(val_ood, cfg.batch_size, device, False),
    )


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for iq, label, _ in loader:
        logits = model(iq)
        correct += (logits.argmax(1) == label).sum().item()
        total += label.numel()
    return correct / total


def _plot_curves(history, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = history["epochs"]
    axes[0].plot(ep, history["train_loss"], color="tab:blue")
    axes[0].set(xlabel="epoch", ylabel="train loss", title=f"{history['run']} -- loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        ep, history["val_acc_indist"], label="in-distribution", color="tab:green"
    )
    axes[1].plot(ep, history["val_acc_ood"], label="full-circle (OOD)", color="tab:red")
    axes[1].axhline(1.0 / len(ModClassConfig().modulations), color="0.6", ls=":", label="chance")
    axes[1].set(
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0, 1.02),
        title=f"{history['run']} -- val accuracy",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_one(run_name, model_name, train_phase_deg, cfg, device, out_root):
    print(
        f"\n{'='*64}\n  Run: {run_name}  (model={model_name}, train theta in +/-{train_phase_deg:.0f} deg)\n{'='*64}"
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model(model_name, cfg).to(device)
    pc = count_parameters(model)
    print(f"  params: {pc['count']:,d} ({pc['real']:,d} real-equiv)")

    train_dl, val_ind_dl, val_ood_dl = make_loaders(cfg, train_phase_deg, device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    history = {
        "run": run_name,
        "model": model_name,
        "train_phase_deg": train_phase_deg,
        "params": pc,
        "epochs": [],
        "train_loss": [],
        "val_acc_indist": [],
        "val_acc_ood": [],
    }
    best_ood = -1.0
    best_epoch = -1

    with open(log_path, "w") as logf:
        logf.write(f"run={run_name} model={model_name} train_phase_deg={train_phase_deg}\nparams={pc}\n\n")
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum = nb = 0
            for iq, label, _ in train_dl:
                opt.zero_grad()
                loss = criterion(model(iq), label)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                loss_sum += loss.item()
                nb += 1
            train_loss = loss_sum / nb

            acc_ind = evaluate(model, val_ind_dl)
            acc_ood = evaluate(model, val_ood_dl)
            dt = time.time() - t0

            line = (
                f"  epoch {epoch:>3d}/{cfg.epochs}  loss={train_loss:.4f}  val_acc[in-dist]={acc_ind:.4f}  "
                f"val_acc[full-circle]={acc_ood:.4f}  ({dt:.1f}s)"
            )
            print(line)
            logf.write(line + "\n")

            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_acc_indist"].append(acc_ind)
            history["val_acc_ood"].append(acc_ood)

            if acc_ood > best_ood:
                best_ood = acc_ood
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    history["summary"] = {
        "best_val_acc_ood": best_ood,
        "best_epoch": best_epoch,
        "final_val_acc_indist": acc_ind,
        "final_val_acc_ood": acc_ood,
        "gap_indist_minus_ood": acc_ind - acc_ood,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        import json

        json.dump(history, f, indent=2)
    _plot_curves(history, out_dir)

    print(f"  best full-circle acc: {best_ood:.4f} (epoch {best_epoch})")
    print(
        f"  final in-dist={acc_ind:.4f}  full-circle={acc_ood:.4f}  gap={acc_ind-acc_ood:+.4f}"
    )
    return history


def cmd_train(args):
    cfg = ModClassConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    out_dir = args.out_dir if args.out_dir is not None else cfg.out_dir
    cfg.out_dir = resolve_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | classes: {cfg.modulations}  | SNR: {cfg.snr_db} dB")
    print(f"Train/Val sizes: {cfg.n_train}/{cfg.n_val}  | epochs: {cfg.epochs}")

    specs = run_specs(cfg)
    summary = {}
    for run in args.runs:
        if run not in specs:
            raise ValueError(f"unknown run '{run}', choices: {list(specs)}")
        model_name, phase = specs[run]
        h = train_one(run, model_name, phase, cfg, device, cfg.out_dir)
        summary[run] = h["summary"]

    print(f"\n{'='*64}\n  Summary\n{'='*64}")
    for run, s in summary.items():
        print(
            f"  {run:16s}  in-dist={s['final_val_acc_indist']:.4f}  full-circle={s['final_val_acc_ood']:.4f}  gap={s['gap_indist_minus_ood']:+.4f}"
        )
