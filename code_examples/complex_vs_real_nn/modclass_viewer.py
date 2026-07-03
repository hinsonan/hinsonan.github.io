"""Interactive Gradio viewer for the complex-vs-real AMC experiment."""
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from modclass_core import (
    ModClassConfig,
    add_awgn,
    build_model,
    constellation,
    generate_clean_burst,
    rotate_burst,
)


RUNS = {
    "complex_narrow": ("complex", "Complex narrow"),
    "complex_moment": ("complex_moment", "Complex+moments"),
    "real_narrow": ("real", "Real narrow"),
    "real_full": ("real", "Real full"),
}

BASE_DIR = Path(__file__).resolve().parent


def load_models(cfg, device):
    models = {}
    warnings = []
    root = BASE_DIR / cfg.out_dir
    for run, (model_name, label) in RUNS.items():
        ckpt = root / run / "best_model.pt"
        if not ckpt.exists():
            continue
        model = build_model(model_name, cfg).to(device)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        except Exception as exc:
            warnings.append(f"{run}: could not load checkpoint ({exc})")
            continue
        model.eval()
        models[run] = (label, model)
    return models, warnings


def clean_seed(seed, default):
    if seed is None:
        return int(default)
    try:
        value = float(seed)
    except (TypeError, ValueError):
        return int(default)
    if np.isnan(value):
        return int(default)
    return int(value)


def make_symbols(modulation, cfg, seed):
    rng = np.random.default_rng(clean_seed(seed, cfg.seed))
    return generate_clean_burst(modulation, cfg, rng)


def rotate_and_noise(symbols, angle_deg, snr_db, seed):
    rng = np.random.default_rng(clean_seed(seed, 0) + 10_000)
    rotated = rotate_burst(symbols, np.deg2rad(angle_deg))
    return add_awgn(rotated, snr_db, rng), rotated


@torch.no_grad()
def predict(models, burst, cfg, device):
    if not models:
        return []
    x = torch.from_numpy(burst[None, :]).to(device)
    rows = []
    for run, (label, model) in models.items():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        rows.append(
            {
                "run": run,
                "model": label,
                "prediction": cfg.modulations[pred_idx],
                **{mod: float(probs[i]) for i, mod in enumerate(cfg.modulations)},
            }
        )
    return rows


def iq_scatter(noisy, modulation, angle_deg, show_reference):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=noisy.real,
            y=noisy.imag,
            mode="markers",
            name="noisy burst",
            marker={"size": 7, "opacity": 0.65},
        )
    )
    if show_reference:
        ref = rotate_burst(constellation(modulation), np.deg2rad(angle_deg))
        fig.add_trace(
            go.Scatter(
                x=ref.real,
                y=ref.imag,
                mode="markers",
                name="clean constellation",
                marker={"size": 14, "symbol": "x", "color": "black"},
            )
        )
    lim = max(1.6, float(np.max(np.abs(noisy))) * 1.15)
    fig.update_layout(
        title="IQ scatter",
        xaxis_title="I",
        yaxis_title="Q",
        xaxis={"range": [-lim, lim], "zeroline": True},
        yaxis={"range": [-lim, lim], "zeroline": True, "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def time_trace(noisy):
    t = np.arange(len(noisy))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=noisy.real, mode="lines", name="I"))
    fig.add_trace(go.Scatter(x=t, y=noisy.imag, mode="lines", name="Q"))
    fig.update_layout(
        title="Time-domain I/Q trace",
        xaxis_title="sample",
        yaxis_title="amplitude",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def probability_plot(rows, cfg):
    fig = go.Figure()
    if not rows:
        fig.add_annotation(
            text="No checkpoints found under trained_modclass/", showarrow=False
        )
    for row in rows:
        fig.add_trace(
            go.Bar(x=list(cfg.modulations), y=[row[m] for m in cfg.modulations], name=row["run"])
        )
    fig.update_layout(
        title="Prediction probabilities",
        xaxis_title="modulation",
        yaxis_title="probability",
        yaxis={"range": [0, 1]},
        barmode="group",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


@torch.no_grad()
def prediction_heatmap(symbols, modulation, snr_db, seed, models, cfg, device):
    """Predicted-class distribution across rotation angles, one heatmap per model.

    For the selected true modulation, runs each model on the same symbol
    sequence rotated to every angle and shows the full softmax vector (all 4
    classes) as a heatmap.  The true class is pinned to the top row, so an
    invariant model shows a single bright top row across all angles while a
    non-invariant model's bright row jumps as the rotation leaves the
    training band (marked with a translucent vertical band).
    """
    angles = np.arange(-180, 181, 10)
    mods = list(cfg.modulations)
    true_idx = mods.index(modulation)
    order = [true_idx] + [i for i in range(len(mods)) if i != true_idx]
    y_labels = [mods[i] for i in order]

    if not models:
        fig = go.Figure()
        fig.add_annotation(
            text="No checkpoints loaded — run `python modclass_cli.py train` first",
            showarrow=False,
        )
        fig.update_layout(
            title=f"Predicted-class distribution vs rotation (true: {modulation.upper()})",
            xaxis_title="rotation angle (deg)",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        return fig

    bursts = np.stack([rotate_and_noise(symbols, a, snr_db, seed)[0] for a in angles])
    x = torch.from_numpy(bursts).to(device)
    n_models = len(models)

    train_phases = {
        "complex_narrow": cfg.train_phase_deg,
        "complex_moment": cfg.train_phase_deg,
        "real_narrow": cfg.train_phase_deg,
        "real_full": cfg.full_phase_deg,
    }

    fig = make_subplots(
        rows=1,
        cols=n_models,
        subplot_titles=[
            f"{label} (trained ±{train_phases.get(run, cfg.train_phase_deg):.0f}°)"
            for run, (label, _) in models.items()
        ],
        shared_yaxes=True,
        horizontal_spacing=0.07,
    )

    for col, (run, (label, model)) in enumerate(models.items(), start=1):
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        z = probs[:, order].T
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=angles,
                y=y_labels,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                showscale=(col == n_models),
                colorbar=dict(title="P(pred)", thickness=12) if col == n_models else None,
            ),
            row=1,
            col=col,
        )
        xref = "x" if col == 1 else f"x{col}"
        band = train_phases.get(run, cfg.train_phase_deg)
        fig.add_vrect(
            x0=-band,
            x1=band,
            fillcolor="white",
            opacity=0.12,
            line_width=0,
            xref=xref,
        )
        fig.update_xaxes(
            title_text="rotation angle (deg)" if col == 1 else None,
            row=1,
            col=col,
        )

    fig.update_layout(
        title=f"Predicted-class distribution vs rotation (true: {modulation.upper()}) — "
              f"top row = true class; white band = training range",
        margin={"l": 40, "r": 70, "t": 60, "b": 40},
    )
    fig.update_yaxes(title_text="predicted class", row=1, col=1)
    return fig


def build_app():
    cfg = ModClassConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, load_warnings = load_models(cfg, device)
    table_headers = ["run", "model", "prediction", *cfg.modulations]
    default_mod = cfg.modulations[0]
    default_angle = 0
    default_snr = cfg.snr_db
    default_seed = cfg.seed
    default_show_ref = True

    def update(modulation, angle_deg, snr_db, seed, show_reference):
        symbols = make_symbols(modulation, cfg, seed)
        noisy, _ = rotate_and_noise(symbols, angle_deg, snr_db, seed)
        rows = predict(models, noisy, cfg, device)
        if not rows:
            rows = [
                {
                    "run": "none",
                    "model": "No checkpoints found",
                    "prediction": "",
                    **{m: np.nan for m in cfg.modulations},
                }
            ]
        pred_rows = rows if rows[0]["run"] != "none" else []
        return (
            iq_scatter(noisy, modulation, angle_deg, show_reference),
            time_trace(noisy),
            pd.DataFrame(rows, columns=table_headers),
            probability_plot(pred_rows, cfg),
            prediction_heatmap(symbols, modulation, snr_db, seed, models, cfg, device),
        )

    initial = update(default_mod, default_angle, default_snr, default_seed, default_show_ref)

    with gr.Blocks(title="Complex vs Real NN Viewer") as demo:
        gr.Markdown("# Complex vs Real NN Interactive Viewer")
        gr.Markdown(f"Loaded checkpoints: {', '.join(models) if models else 'none'}")
        if load_warnings:
            gr.Markdown("\n".join(f"- {warning}" for warning in load_warnings))
        with gr.Row():
            modulation = gr.Dropdown(list(cfg.modulations), value=default_mod, label="Modulation")
            angle = gr.Slider(-180, 180, value=default_angle, step=1, label="Rotation angle (deg)")
            snr = gr.Slider(-10, 30, value=default_snr, step=1, label="SNR (dB)")
            seed = gr.Number(value=default_seed, precision=0, label="Seed")
            show_ref = gr.Checkbox(value=default_show_ref, label="Show clean constellation reference")
        with gr.Row():
            scatter = gr.Plot(label="IQ scatter", value=initial[0])
            trace = gr.Plot(label="I/Q trace", value=initial[1])
        with gr.Row():
            table = gr.Dataframe(
                label="Prediction probabilities",
                interactive=False,
                value=initial[2],
            )
            probs = gr.Plot(label="Probability bars", value=initial[3])
        sweep = gr.Plot(label="Prediction distribution vs rotation", value=initial[4])

        inputs = [modulation, angle, snr, seed, show_ref]
        outputs = [scatter, trace, table, probs, sweep]
        for control in inputs:
            control.change(update, inputs, outputs)
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
