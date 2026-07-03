"""Evaluation logic for the AMC experiment."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from config import ModClassConfig
from data import generate_dataset
from models import count_parameters
from plotting import (
    plot_confusion,
    plot_per_modulation,
    plot_rotation_generalization,
    plot_snr_sweep,
)
from training import RUNS, resolve_dir


def load_run(run_name, cfg, model_dir, device):
    from models import build_model

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
        raise SystemExit("No checkpoints found -- run `python train.py train` first.")
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
