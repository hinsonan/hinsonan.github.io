"""Visualization helpers for the AMC experiment."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .data import CONSTELLATIONS, add_awgn, generate_burst, generate_clean_burst, rotate_burst
    from .training import RUNS
except ImportError:
    from data import CONSTELLATIONS, add_awgn, generate_burst, generate_clean_burst, rotate_burst
    from training import RUNS


def _square_axes(ax, lim):
    """Set square axes with equal aspect and grid lines.

    Args:
        ax: Matplotlib Axes object.
        lim: Half-width of both x and y limits.
    """
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.grid(True, alpha=0.25)


def plot_constellations(cfg, out_dir):
    """Save a grid showing ideal constellations alongside noisy, rotated
    received bursts for each modulation class.

    Args:
        cfg: Experiment configuration (defines modulations, SNR, etc.).
        out_dir: Directory to save the figure in.
    """
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
    """Save a figure illustrating how rotation changes I/Q appearance.

    Shows the same clean burst rotated by 0, 30, 60, 90 degrees with
    added noise — same label but very different I/Q patterns.

    Args:
        cfg: Experiment configuration.
        out_dir: Directory to save the figure in.
        mod: Modulation scheme to demonstrate (default 'qpsk').
    """
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


def plot_rotation_generalization(angles, sweep, cfg, out_path):
    """Plot accuracy vs rotation angle for all trained runs.

    Args:
        angles: Array of rotation angles (degrees) used for evaluation.
        sweep: Dictionary mapping run names to accuracy arrays.
        cfg: Experiment configuration (provides training band, n_classes).
        out_path: Path to save the figure.
    """
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
    """Plot per-modulation accuracy vs rotation angle.

    Highlights how each constellation's rotational symmetry affects
    real-network performance.

    Args:
        angles: Array of rotation angles (degrees).
        per_mod: Dictionary mapping run names to lists of per-class
            accuracy arrays.
        cfg: Experiment configuration.
        out_path: Path to save the figure.
    """
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
    """Plot full-circle accuracy vs SNR for all trained runs.

    Args:
        snrs: Array of SNR values (dB).
        snr_acc: Dictionary mapping run names to accuracy arrays.
        cfg: Experiment configuration.
        out_path: Path to save the figure.
    """
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
    """Save a confusion matrix grid across rotation angles for key runs.

    Args:
        cms: Dictionary mapping run names to dicts of
            ``{angle: confusion_matrix}``.
        cfg: Experiment configuration.
        out_path: Path to save the figure.
    """
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
