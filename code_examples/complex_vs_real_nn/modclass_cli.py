"""CLI for training, evaluation, and visualization for the AMC experiment."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amc.config import ModClassConfig
from amc.data import (
    CONSTELLATIONS,
    add_awgn,
    generate_burst,
    generate_clean_burst,
    generate_dataset,
    rotate_burst,
)
from amc.models import build_model, count_parameters

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


def load_run(run_name, cfg, model_dir, device):
    model_name = RUNS[run_name][0]
    ckpt = os.path.join(model_dir, run_name, "best_model.pt")
    if not os.path.exists(ckpt):
        return None
    model = build_model(model_name, cfg).to(device)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except Exception as exc:
        print(f"  skipped {run_name}: could not load checkpoint ({exc})")
        return None
    model.eval()
    return model


@torch.no_grad()
def accuracy_at_angle(model, cfg, angle_deg, snr_db, device, n=2000, per_class=False):
    data = generate_dataset(n, cfg, angle_deg, angle_deg, snr_db=snr_db, seed_offset=900)
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    pred = model(iq).argmax(1)
    if per_class:
        n_cls = cfg.n_classes
        accs = []
        for c in range(n_cls):
            m = label == c
            accs.append((pred[m] == label[m]).float().mean().item() if m.any() else float("nan"))
        return accs
    return (pred == label).float().mean().item()


@torch.no_grad()
def confusion_at_angle(model, cfg, angle_deg, snr_db, device, n=4000):
    data = generate_dataset(n, cfg, angle_deg, angle_deg, snr_db=snr_db, seed_offset=901)
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    pred = model(iq).argmax(1)
    k = cfg.n_classes
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(label.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm


@torch.no_grad()
def accuracy_full_circle(model, cfg, snr_db, device, n=4000):
    data = generate_dataset(
        n, cfg, -cfg.full_phase_deg, cfg.full_phase_deg, snr_db=snr_db, seed_offset=902
    )
    iq = torch.from_numpy(data["iq"]).to(device)
    label = torch.from_numpy(data["label"]).to(device)
    return (model(iq).argmax(1) == label).float().mean().item()


def plot_rotation_generalization(angles, sweep, cfg, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, (_, color, label) in RUNS.items():
        if run in sweep:
            ax.plot(angles, sweep[run], color=color, label=label, lw=2)
    ax.axvspan(
        -cfg.train_phase_deg,
        cfg.train_phase_deg,
        color="0.85",
        label=f"training band (+/-{cfg.train_phase_deg:.0f} deg)",
    )
    ax.axhline(1.0 / cfg.n_classes, color="0.5", ls=":", lw=1, label="chance")
    ax.set_xlabel("test rotation angle (degrees)")
    ax.set_ylabel("classification accuracy")
    ax.set_title(
        "Rotation generalization: accuracy vs unseen carrier phase\n"
        f"{cfg.n_classes}-class modulation classification @ {cfg.snr_db:.0f} dB SNR"
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlim(angles[0], angles[-1])
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_modulation(angles, per_mod, cfg, out_path):
    mods = list(cfg.modulations)
    fig, axes = plt.subplots(1, len(mods), figsize=(4 * len(mods), 3.6), sharey=True)
    for mi, (mod, ax) in enumerate(zip(mods, axes)):
        for run in ("complex_moment", "complex_narrow", "real_narrow"):
            if run in per_mod:
                _, color, label = RUNS[run]
                ax.plot(angles, [a[mi] for a in per_mod[run]], color=color, lw=1.8, label=label)
        ax.axvspan(-cfg.train_phase_deg, cfg.train_phase_deg, color="0.85")
        ax.set_title(mod.upper())
        ax.set_xlabel("rotation (deg)")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.set_ylabel("accuracy")
            ax.legend(fontsize=7, loc="lower center")
    fig.suptitle(
        "Per-modulation accuracy vs rotation -- the real net's dips trace each constellation's rotational symmetry",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_snr_sweep(snrs, snr_acc, cfg, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, (_, color, label) in RUNS.items():
        if run in snr_acc:
            ax.plot(snrs, snr_acc[run], "-o", color=color, label=label, ms=4)
    ax.axhline(1.0 / cfg.n_classes, color="0.5", ls=":", label="chance")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy (full-circle test rotations)")
    ax.set_title("Modulation classification vs SNR\n(test rotations span the full circle)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_confusion(cms, cfg, out_path):
    runs = [r for r in ("complex_moment", "complex_narrow", "real_narrow") if r in cms]
    if not runs:
        print("  skipped confusion plot (needs a narrow-trained complex or real model)")
        return
    angles = list(cms[runs[0]].keys())
    fig, axes = plt.subplots(len(runs), len(angles), figsize=(3.2 * len(angles), 3.2 * len(runs)))
    axes = np.atleast_2d(axes)
    mods = [m.upper() for m in cfg.modulations]
    for ri, run in enumerate(runs):
        for ci, ang in enumerate(angles):
            ax = axes[ri, ci]
            cm = cms[run][ang]
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(mods)), mods, fontsize=7, rotation=45)
            ax.set_yticks(range(len(mods)), mods, fontsize=7)
            acc = np.trace(cm) / cm.sum()
            tag = "in-band" if abs(ang) <= cfg.train_phase_deg else "OOD"
            ax.set_title(f"{RUNS[run][2]}\n{ang:.0f} deg ({tag}), acc={acc:.2f}", fontsize=8)
            for i in range(len(mods)):
                for j in range(len(mods)):
                    ax.text(
                        j,
                        i,
                        f"{cmn[i,j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black",
                    )
    fig.suptitle("Confusion matrices: in-band vs out-of-distribution rotation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def cmd_eval(args):
    model_dir = resolve_dir(args.model_dir)
    results_dir = resolve_dir(args.results_dir)
    viz_dir = resolve_dir(args.viz_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModClassConfig()
    print(f"Device: {device}  classes={cfg.modulations}")

    models = {r: load_run(r, cfg, model_dir, device) for r in RUNS}
    models = {r: m for r, m in models.items() if m is not None}
    if not models:
        raise SystemExit("No checkpoints found -- run `python modclass_cli.py train` first.")
    for r, m in models.items():
        print(f"  loaded {r}: {count_parameters(m)}")

    angles = np.arange(-180, 181, 5.0)
    sweep, per_mod = {}, {}
    for r, m in models.items():
        sweep[r] = [accuracy_at_angle(m, cfg, a, cfg.snr_db, device, args.n) for a in angles]
        per_mod[r] = [
            accuracy_at_angle(m, cfg, a, cfg.snr_db, device, args.n, per_class=True)
            for a in angles
        ]
        print(f"  rotation sweep done: {r} (min={min(sweep[r]):.3f}, max={max(sweep[r]):.3f})")

    snrs = [20, 15, 10, 5, 0, -5, -10]
    snr_acc = {
        r: [accuracy_full_circle(m, cfg, s, device, 4000) for s in snrs]
        for r, m in models.items()
    }

    angle_in, angle_ood = 0.0, 90.0
    cms = {
        r: {
            angle_in: confusion_at_angle(m, cfg, angle_in, cfg.snr_db, device),
            angle_ood: confusion_at_angle(m, cfg, angle_ood, cfg.snr_db, device),
        }
        for r, m in models.items()
    }

    results = {
        "angles_deg": angles.tolist(),
        "snrs_db": snrs,
        "config": {
            "modulations": list(cfg.modulations),
            "train_phase_deg": cfg.train_phase_deg,
            "snr_db": cfg.snr_db,
        },
        "rotation_sweep": {r: sweep[r] for r in sweep},
        "snr_sweep": {r: snr_acc[r] for r in snr_acc},
        "params": {r: count_parameters(m) for r, m in models.items()},
        "summary": {
            r: {
                "acc_in_band_0deg": sweep[r][list(angles).index(0.0)],
                "acc_at_90deg": sweep[r][list(angles).index(90.0)],
                "acc_min_over_circle": min(sweep[r]),
                "acc_mean_over_circle": float(np.mean(sweep[r])),
            }
            for r in sweep
        },
    }
    out_json = os.path.join(results_dir, "rotation_sweep.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out_json}")

    plot_rotation_generalization(angles, sweep, cfg, os.path.join(viz_dir, "rotation_generalization.png"))
    plot_per_modulation(angles, per_mod, cfg, os.path.join(viz_dir, "rotation_per_modulation.png"))
    plot_snr_sweep(snrs, snr_acc, cfg, os.path.join(viz_dir, "snr_sweep_modclass.png"))
    plot_confusion(cms, cfg, os.path.join(viz_dir, "confusion_in_vs_ood.png"))

    print(f"\n{'='*64}\n  Rotation-generalization summary\n{'='*64}")
    for r in sweep:
        s = results["summary"][r]
        print(
            f"  {r:16s}  acc@0deg={s['acc_in_band_0deg']:.3f}  acc@90deg={s['acc_at_90deg']:.3f}  min={s['acc_min_over_circle']:.3f}  mean={s['acc_mean_over_circle']:.3f}"
        )


def _square_axes(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.grid(True, alpha=0.25)


def plot_constellations(cfg, out_dir):
    mods = list(cfg.modulations)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, len(mods), figsize=(3.2 * len(mods), 6.4))

    for ci, mod in enumerate(mods):
        pts = CONSTELLATIONS[mod]

        ax = axes[0, ci]
        ax.scatter(pts.real, pts.imag, s=70, color="tab:blue", edgecolors="k", zorder=3)
        ax.set_title(f"{mod.upper()}  ({len(pts)} pts)", fontsize=11)
        _square_axes(ax, lim=1.8)
        if ci == 0:
            ax.set_ylabel("ideal\nImaginary", fontsize=10)

        ax = axes[1, ci]
        burst, theta = generate_burst(
            mod, cfg, rng, -cfg.full_phase_deg, cfg.full_phase_deg, cfg.snr_db
        )
        ax.scatter(burst.real, burst.imag, s=8, alpha=0.35, color="tab:red")
        ax.set_title(f"rotated {np.rad2deg(theta):+.0f} deg @ {cfg.snr_db:.0f} dB", fontsize=9)
        _square_axes(ax, lim=2.4)
        ax.set_xlabel("Real", fontsize=10)
        if ci == 0:
            ax.set_ylabel("received\nImaginary", fontsize=10)

    fig.suptitle(
        "Modulation classification: name the scheme from the noisy, rotated burst (bottom row)",
        y=1.0,
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "modclass_constellations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_rotation_nuisance(cfg, out_dir, mod="qpsk"):
    angles = [0, 30, 60, 90]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, len(angles), figsize=(3.0 * len(angles), 3.2))

    burst = generate_clean_burst(mod, cfg, rng)
    for ax, ang in zip(axes, angles):
        sym = add_awgn(rotate_burst(burst, np.deg2rad(ang)), cfg.snr_db, rng)
        ax.scatter(sym.real, sym.imag, s=12, alpha=0.5, color="tab:purple")
        ax.set_title(f"{ang} deg rotation", fontsize=10)
        _square_axes(ax, lim=1.8)
        ax.set_xlabel("Real", fontsize=9)
    axes[0].set_ylabel("Imaginary", fontsize=9)
    fig.suptitle(
        f"Same label ({mod.upper()}), different I/Q pattern at each carrier phase -- rotation is a nuisance variable",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "modclass_rotation_nuisance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def cmd_viz(args):
    cfg = ModClassConfig()
    viz_dir = resolve_dir(args.viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    plot_constellations(cfg, viz_dir)
    plot_rotation_nuisance(cfg, viz_dir)


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")

    ap_train = sub.add_parser("train")
    ap_train.add_argument(
        "--runs",
        nargs="+",
        default=["complex_moment", "complex_narrow", "real_narrow", "real_full"],
    )
    ap_train.add_argument("--epochs", type=int, default=None)
    ap_train.add_argument("--out_dir", type=str, default=None)
    ap_train.set_defaults(func=cmd_train)

    ap_eval = sub.add_parser("eval")
    ap_eval.add_argument("--model_dir", default="trained_modclass")
    ap_eval.add_argument("--results_dir", default="results")
    ap_eval.add_argument("--viz_dir", default="visualizations")
    ap_eval.add_argument("--n", type=int, default=2000)
    ap_eval.set_defaults(func=cmd_eval)

    ap_viz = sub.add_parser("viz")
    ap_viz.add_argument("--viz_dir", default="visualizations")
    ap_viz.set_defaults(func=cmd_viz)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
