"""Plot builders for the Part 1 notebook visuals."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from complex_core import (
    CONSTELLATIONS,
    MODULATION_NOTES,
    cauchy_riemann_residual,
    complex_map_matrix,
    complex_linear_mse_gradients,
    complex_sinusoid,
    iq_impairment_examples,
    magnitude,
    make_complex_regression_data,
    mul_by_i,
    phase,
    phase_shift_same_signal,
    rotation_commutation_error_complex,
    rotation_commutation_error_real,
    run_iq_rotation_sample_efficiency,
    rotate,
    sample_constellation_burst,
    simulate_descent_methods,
    toy_quadratic_gradients,
    toy_quadratic_loss,
    to_wirtinger,
    train_complex_linear,
    wirtinger_steps,
)


COLORS = {
    "original": "#1f77b4",
    "rotated": "#ff7f0e",
    "i_mul": "#2ca02c",
    "clean": "#1f77b4",
    "noisy": "#d62728",
}


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        autosize=True,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
    )
    return fig


def plot_rotation_vectors(x: float, y: float, angle_deg: float) -> go.Figure:
    z = np.array([x + 1j * y])
    z_i = mul_by_i(z)
    z_theta = rotate(z, angle_deg)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, z.real[0]], y=[0, z.imag[0]], mode="lines+markers", name="z", line={"color": COLORS["original"], "width": 3}))
    fig.add_trace(go.Scatter(x=[0, z_i.real[0]], y=[0, z_i.imag[0]], mode="lines+markers", name="i*z", line={"color": COLORS["i_mul"], "width": 3, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=[0, z_theta.real[0]], y=[0, z_theta.imag[0]], mode="lines+markers", name=f"e^(i {angle_deg} deg) z", line={"color": COLORS["rotated"], "width": 3, "dash": "dot"}))

    lim = max(1.5, abs(z[0]) * 1.3, abs(z_theta[0]) * 1.3)
    fig.add_shape(type="circle", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"color": "gray", "dash": "dot"})
    fig.update_xaxes(title_text="Real", range=[-lim, lim], zeroline=True)
    fig.update_yaxes(title_text="Imaginary", range=[-lim, lim], zeroline=True, scaleanchor="x", scaleratio=1)
    return _layout(fig, "Complex rotation: i*z and e^(i*theta)*z")


def format_wirtinger_steps_markdown(example_name: str, x: float, y: float) -> str:
    s = wirtinger_steps(example_name, x + 1j * y)
    return (
        f"**Given:** `z = {s['x']:.3f} + ({s['y']:.3f})i = {s['z']:.3f}`\n\n"
        f"**Function:** `f(z) = {s['f_expr']}`\n\n"
        f"**Step 1:** `f(z) = {s['f_val']:.4f}`\n\n"
        f"**Step 2:** `df/dx = {s['df_dx']:.4f}`, `df/dy = {s['df_dy']:.4f}`\n\n"
        "**Step 3:**\n"
        f"`df/dz  = 0.5*(df/dx - i*df/dy) = {s['df_dz']:.4f}`\n\n"
        f"`df/dz* = 0.5*(df/dx + i*df/dy) = {s['df_dz_conj']:.4f}`\n"
    )


def format_holomorphic_check_markdown(z: complex = 1.2 - 0.7j) -> str:
    """Return a compact Cauchy-Riemann comparison for notebook display."""
    rows = []
    labels = {
        "z2": "`f(z)=z^2`",
        "conj_z": "`f(z)=z*`",
        "abs2_as_complex": "`f(z)=|z|^2`",
    }
    notes = {
        "z2": "holomorphic: ordinary complex derivative works",
        "conj_z": "anti-holomorphic: depends on `z*`",
        "abs2_as_complex": "real loss shape: depends on both `z` and `z*`",
    }
    for name in ["z2", "conj_z", "abs2_as_complex"]:
        r = cauchy_riemann_residual(name, z)
        residual = abs(r["residual_1"]) + abs(r["residual_2"])
        rows.append(f"| {labels[name]} | `{residual:.2e}` | {notes[name]} |")

    return "\n".join(
        [
            f"Cauchy-Riemann residuals at `z={z:.3f}`:",
            "",
            "| Function | residual | Interpretation |",
            "|---|---:|---|",
            *rows,
            "",
            "Small residual means the ordinary complex derivative is valid. ML losses like `|z|^2` are real-valued and non-holomorphic, so Wirtinger calculus is the useful tool.",
        ]
    )


def plot_backprop_training(
    n_samples: int = 128,
    lr: float = 0.1,
    epochs: int = 50,
    noise: float = 0.1,
    seed: int = 7,
    w_init: complex = 0.0 + 0.0j,
    b_init: complex = 0.0 + 0.0j,
) -> tuple[go.Figure, str, dict]:
    x, y, w_true, b_true = make_complex_regression_data(n_samples=n_samples, noise=noise, seed=seed)
    history = train_complex_linear(x, y, lr=lr, epochs=epochs, w_init=w_init, b_init=b_init, seed=seed)

    w_final = history["w"][-1]
    b_final = history["b"][-1]
    gw, _, _ = complex_linear_mse_gradients(x, y, w_final, b_final)

    eps = 1e-5

    def loss_fn(uw, vw, ub, vb):
        wp = uw + 1j * vw
        bp = ub + 1j * vb
        return np.mean(np.abs(wp * x + bp - y) ** 2)

    l0 = loss_fn(w_final.real, w_final.imag, b_final.real, b_final.imag)
    l_wu = loss_fn(w_final.real + eps, w_final.imag, b_final.real, b_final.imag)
    l_wv = loss_fn(w_final.real, w_final.imag + eps, b_final.real, b_final.imag)
    _, gw_fd = to_wirtinger((l_wu - l0) / eps, (l_wv - l0) / eps)
    err = abs(gw - gw_fd)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss curve", "w trajectory in complex plane"), horizontal_spacing=0.16)
    fig.add_trace(go.Scatter(x=list(range(len(history["loss"]))), y=history["loss"], mode="lines+markers", name="loss"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[w.real for w in history["w"]],
            y=[w.imag for w in history["w"]],
            mode="lines+markers",
            name="w",
            marker={"size": 6, "color": history["loss"], "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "loss", "x": 0.43, "len": 0.8}},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=[w_true.real], y=[w_true.imag], mode="markers", name="true w", marker={"size": 14, "symbol": "x", "color": "red"}), row=1, col=2)
    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_yaxes(title_text="MSE loss", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Re(w)", row=1, col=2)
    fig.update_yaxes(title_text="Im(w)", row=1, col=2)
    fig = _layout(fig, "Manual complex backprop")

    summary = (
        f"True: w={w_true:.3f}, b={b_true:.3f}\n"
        f"Learned: w={w_final:.3f}, b={b_final:.3f}\n"
        f"Final loss: {history['loss'][-1]:.4e}\n"
        f"Gradient FD check error on w: {err:.3e}"
    )
    stats = {"w_true": w_true, "b_true": b_true, "w_final": w_final, "b_final": b_final, "grad_err": err}
    return fig, summary, stats


def plot_sinusoid_and_iq(
    amplitude: float = 1.0,
    frequency: float = 2.0,
    phase_deg: float = 0.0,
    snr_db: float | None = 30.0,
    shift_deg: float = 45.0,
    seed: int = 0,
    n: int = 128,
) -> go.Figure:
    z1, t = complex_sinusoid(n, amplitude, frequency, phase_deg, snr_db, seed)
    z2 = phase_shift_same_signal(z1, shift_deg)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("I(t) and Q(t)", "IQ trajectory"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=t, y=z1.real, mode="lines", name="I", line={"color": "#1f77b4"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=z1.imag, mode="lines", name="Q", line={"color": "#ff7f0e"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=z1.real, y=z1.imag, mode="markers", name="original", marker={"size": 4, "color": COLORS["original"]}), row=2, col=1)
    fig.add_trace(go.Scatter(x=z2.real, y=z2.imag, mode="markers", name=f"+{shift_deg} deg phase", marker={"size": 4, "color": COLORS["rotated"]}), row=2, col=1)
    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_yaxes(title_text="amplitude", row=1, col=1)
    fig.update_xaxes(title_text="I", row=2, col=1)
    fig.update_yaxes(title_text="Q", row=2, col=1)
    return _layout(fig, "Complex sinusoid and global phase shift")


def plot_constellation_samples(
    mod_name: str = "qpsk",
    phase_deg: float = 30.0,
    snr_db: float | None = 20.0,
    n_symbols: int = 256,
    seed: int = 0,
) -> tuple[go.Figure, str]:
    if mod_name not in CONSTELLATIONS:
        raise ValueError(f"Unknown modulation '{mod_name}'. Options: {list(CONSTELLATIONS)}")

    burst, clean = sample_constellation_burst(mod_name, n_symbols, phase_deg, snr_db, seed)
    ideal = clean * np.exp(1j * np.deg2rad(phase_deg))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=burst.real, y=burst.imag, mode="markers", name="received", marker={"size": 6, "color": COLORS["noisy"], "opacity": 0.7}))
    fig.add_trace(go.Scatter(x=ideal.real, y=ideal.imag, mode="markers", name="ideal symbols", marker={"size": 12, "symbol": "x", "color": COLORS["clean"]}))
    fig.add_vline(x=0, line={"color": "gray", "dash": "dot"})
    fig.add_hline(y=0, line={"color": "gray", "dash": "dot"})
    lim = max(2.0, float(np.max(np.abs(burst))) * 1.15)
    fig.update_xaxes(title_text="I", range=[-lim, lim], zeroline=True)
    fig.update_yaxes(title_text="Q", range=[-lim, lim], zeroline=True, scaleanchor="x", scaleratio=1)
    fig = _layout(fig, f"{mod_name.upper()} symbols with {phase_deg} deg global phase")

    info = (
        f"Modulation: {mod_name.upper()}\n"
        f"Symbols: {n_symbols}\n"
        f"Avg magnitude: {float(np.mean(magnitude(burst))):.3f}\n"
        f"Avg phase: {float(np.mean(phase(burst, deg=True))):.1f} deg"
    )
    return fig, info


def plot_constellation_rotation_grid(
    rotations_deg: tuple[float, ...] = (0.0, 45.0, 90.0),
    snr_db: float | None = 20.0,
    n_symbols: int = 256,
    seed: int = 0,
) -> go.Figure:
    """Show all modulation classes across multiple global rotations."""
    mods = list(CONSTELLATIONS.keys())
    rotations = list(rotations_deg)
    fig = make_subplots(
        rows=len(mods),
        cols=len(rotations),
        subplot_titles=[f"{m.upper()} @ {r:.0f} deg" for m in mods for r in rotations],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    for row, mod in enumerate(mods, start=1):
        for col, rot in enumerate(rotations, start=1):
            burst, clean = sample_constellation_burst(
                mod,
                n_symbols=n_symbols,
                phase_deg=rot,
                snr_db=snr_db,
                seed=seed + row * 101 + col * 17,
            )
            ideal = clean * np.exp(1j * np.deg2rad(rot))
            show_legend = row == 1 and col == 1
            fig.add_trace(
                go.Scatter(
                    x=burst.real,
                    y=burst.imag,
                    mode="markers",
                    name="received",
                    showlegend=show_legend,
                    marker={"size": 4, "color": COLORS["noisy"], "opacity": 0.65},
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=ideal.real,
                    y=ideal.imag,
                    mode="markers",
                    name="ideal symbols",
                    showlegend=show_legend,
                    marker={"size": 8, "symbol": "x", "color": COLORS["clean"]},
                ),
                row=row,
                col=col,
            )

            lim = max(2.0, float(np.max(np.abs(burst))) * 1.1)
            fig.update_xaxes(range=[-lim, lim], zeroline=True, row=row, col=col)
            fig.update_yaxes(range=[-lim, lim], zeroline=True, row=row, col=col)

    fig.update_layout(
        title="All modulation classes under different global rotations",
        autosize=True,
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        height=300 * len(mods),
    )
    return fig


def modulation_notes_markdown() -> str:
    """Return concise notes on what each class is and where used."""
    lines = ["### Signal classes used in Part 2", ""]
    for key in ["bpsk", "qpsk", "8psk", "16qam"]:
        note = MODULATION_NOTES[key]
        lines.append(f"- **{note['name']}**: {note['what']} {note['uses']}")
    return "\n".join(lines)


def format_wirtinger_update_table_markdown(
    w: complex = 1.0 + 1.5j,
    target: complex = 1.0 + 2.0j,
    lr: float = 0.25,
) -> str:
    """Return a concrete one-step update table for the toy real loss."""
    dL_dw, dL_dw_conj, dL_du, dL_dv = toy_quadratic_gradients(w, target)
    split_grad = 0.5 * dL_du + 0.5j * dL_dv
    correct_next = w - lr * dL_dw_conj
    split_next = w - lr * split_grad
    wrong_next = w - lr * dL_dw

    return "\n".join(
        [
            f"For `L(w)=|w-a|^2`, use `w={w:.3f}`, target `a={target:.3f}`, and `lr={lr:.2f}`.",
            "",
            "| Quantity | Value | Meaning |",
            "|---|---:|---|",
            f"| `w-a` | `{w - target:.3f}` | current error vector |",
            f"| `dL/du` | `{dL_du:.3f}` | real-axis slope |",
            f"| `dL/dv` | `{dL_dv:.3f}` | imaginary-axis slope |",
            f"| `dL/dw*` | `{dL_dw_conj:.3f}` | complex form of the real gradient, scaled by 1/2 |",
            f"| `dL/dw` | `{dL_dw:.3f}` | conjugated direction; not the descent update for real losses |",
            "",
            "One gradient step:",
            "",
            "| Update | New `w` | Loss after step |",
            "|---|---:|---:|",
            f"| `w - lr*dL/dw*` | `{correct_next:.3f}` | `{toy_quadratic_loss(correct_next, target):.4f}` |",
            f"| split real/imag | `{split_next:.3f}` | `{toy_quadratic_loss(split_next, target):.4f}` |",
            f"| `w - lr*dL/dw` | `{wrong_next:.3f}` | `{toy_quadratic_loss(wrong_next, target):.4f}` |",
            "",
            "The first two rows match because `dL/dw* = 0.5*dL/du + 0.5i*dL/dv`. The last row flips the imaginary part of the direction, so it moves the parameter away from the target vertically in this example.",
        ]
    )


def plot_wirtinger_update_directions(
    w: complex = 1.0 + 1.5j,
    target: complex = 1.0 + 2.0j,
    lr: float = 0.25,
) -> tuple[go.Figure, str, dict]:
    """Show a simple one-step comparison for dL/dw* versus dL/dw."""
    dL_dw, dL_dw_conj, dL_du, dL_dv = toy_quadratic_gradients(w, target)
    split_step = -0.5 * lr * dL_du + 1j * (-0.5 * lr * dL_dv)
    correct_step = -lr * dL_dw_conj
    wrong_step = -lr * dL_dw
    split_next = w + split_step
    correct_next = w + correct_step
    wrong_next = w + wrong_step

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Correct update: split-real equals -dL/dw*",
            "Wrong update: dL/dw flips the Q step",
        ),
        horizontal_spacing=0.14,
    )

    def add_points(col: int) -> None:
        fig.add_trace(
            go.Scatter(
                x=[target.real],
                y=[target.imag],
                mode="markers+text",
                name="target a" if col == 1 else "target a ",
                text=["target a"],
                textposition="top center",
                marker={"size": 14, "symbol": "x", "color": "black"},
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[w.real],
                y=[w.imag],
                mode="markers+text",
                name="current w" if col == 1 else "current w ",
                text=["current w"],
                textposition="bottom center",
                marker={"size": 11, "color": "#111111"},
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )

    def add_segment(col: int, start: complex, end: complex, name: str, color: str, dash: str | None = None) -> None:
        fig.add_trace(
            go.Scatter(
                x=[start.real, end.real],
                y=[start.imag, end.imag],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 4, **({"dash": dash} if dash else {})},
                marker={"size": [5, 10], "color": color},
                showlegend=True,
            ),
            row=1,
            col=col,
        )

    add_points(1)
    add_segment(1, w, w + split_step.real, "real-part step", "#2ca02c", dash="dot")
    add_segment(1, w + split_step.real, split_next, "imag-part step", "#2ca02c", dash="dash")
    add_segment(1, w, correct_next, "single complex step -lr*dL/dw*", "#1f77b4")
    fig.add_annotation(
        x=correct_next.real,
        y=correct_next.imag,
        text="same landing point",
        showarrow=True,
        arrowhead=2,
        ax=50,
        ay=-35,
        row=1,
        col=1,
    )

    add_points(2)
    add_segment(2, w, correct_next, "correct: moves toward target", "#1f77b4")
    add_segment(2, w, wrong_next, "wrong: -lr*dL/dw", "#d62728")
    fig.add_annotation(
        x=wrong_next.real,
        y=wrong_next.imag,
        text="Q update goes the wrong way",
        showarrow=True,
        arrowhead=2,
        ax=30,
        ay=45,
        row=1,
        col=2,
    )

    span = 0.9
    for col in (1, 2):
        fig.update_xaxes(title_text="I = Re(w)", range=[target.real - span, target.real + span], zeroline=True, row=1, col=col)
        fig.update_yaxes(title_text="Q = Im(w)", range=[target.imag - span, target.imag + span], zeroline=True, scaleanchor=f"x{col}", scaleratio=1, row=1, col=col)
        fig.add_hline(y=target.imag, line={"color": "#cccccc", "dash": "dot"}, row=1, col=col)
        fig.add_vline(x=target.real, line={"color": "#cccccc", "dash": "dot"}, row=1, col=col)

    fig.update_layout(height=460)
    fig = _layout(fig, "Wirtinger update as split-real gradient descent")

    current_loss = toy_quadratic_loss(w, target)
    correct_loss = toy_quadratic_loss(correct_next, target)
    wrong_loss = toy_quadratic_loss(wrong_next, target)
    summary = (
        f"Current loss={current_loss:.4f}. "
        f"Correct step loss={correct_loss:.4f}; wrong dL/dw step loss={wrong_loss:.4f}. "
        "The left panel shows that split-real and -dL/dw* land at the same point."
    )
    stats = {
        "dL_dw": dL_dw,
        "dL_dw_conj": dL_dw_conj,
        "dL_du": dL_du,
        "dL_dv": dL_dv,
        "next_points": {
            "Correct: -lr dL/dw*": correct_next,
            "Split-real: -(lr/2)(dL/du + i dL/dv)": split_next,
            "Wrong: -lr dL/dw": wrong_next,
        },
    }
    return fig, summary, stats


def plot_descent_trajectory_comparison(
    target: complex = 1.0 + 2.0j,
    w_init: complex = -2.0 + 1.5j,
    lr: float = 0.25,
    epochs: int = 20,
) -> tuple[go.Figure, str, dict]:
    """Compare correct and incorrect complex gradient updates on a toy loss."""
    history = simulate_descent_methods(target=target, w_init=w_init, lr=lr, epochs=epochs)
    colors = {
        "wirtinger_conj": "#1f77b4",
        "real_split": "#2ca02c",
        "wrong_dz": "#d62728",
    }
    labels = {
        "wirtinger_conj": "Wirtinger dL/dw* (correct)",
        "real_split": "Split real-imag (equivalent)",
        "wrong_dz": "Using dL/dw (wrong)",
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Parameter trajectory in complex plane", "Loss vs iteration"),
        horizontal_spacing=0.15,
    )

    for mode in ("wirtinger_conj", "real_split", "wrong_dz"):
        ws = history[mode]["w"]
        x_vals = [w.real for w in ws]
        y_vals = [w.imag for w in ws]
        iter_vals = list(range(len(history[mode]["loss"])))

        if mode == "real_split":
            mode_style = {
                "line": {"width": 2.2, "dash": "dash", "color": "rgba(44,160,44,0.75)"},
                "marker": {
                    "size": 9,
                    "symbol": "diamond-open",
                    "line": {"width": 2.5, "color": colors[mode]},
                    "color": "rgba(0,0,0,0)",
                },
            }
        else:
            mode_style = {
                "line": {"width": 2.5, "color": colors[mode]},
                "marker": {"size": 5, "color": colors[mode]},
            }
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=labels[mode],
                marker=mode_style["marker"],
                line=mode_style["line"],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=iter_vals,
                y=history[mode]["loss"],
                mode="lines+markers",
                name=labels[mode],
                showlegend=False,
                marker=mode_style["marker"],
                line=mode_style["line"],
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=[target.real],
            y=[target.imag],
            mode="markers",
            name="target",
            marker={"size": 14, "symbol": "x", "color": "black"},
        ),
        row=1,
        col=1,
    )

    good_ws = np.array(history["wirtinger_conj"]["w"] + history["real_split"]["w"] + [target], dtype=np.complex128)
    x_min = float(np.min(good_ws.real))
    x_max = float(np.max(good_ws.real))
    y_min = float(np.min(good_ws.imag))
    y_max = float(np.max(good_ws.imag))
    x_pad = max(0.4, 0.22 * max(1e-9, x_max - x_min))
    y_pad = max(0.4, 0.22 * max(1e-9, y_max - y_min))

    fig.update_xaxes(title_text="Re(w)", row=1, col=1)
    fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=1, col=1)
    fig.update_yaxes(
        title_text="Im(w)",
        range=[y_min - y_pad, y_max + y_pad],
        row=1,
        col=1,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_xaxes(title_text="iteration", row=1, col=2)
    fig.update_yaxes(title_text="L=|w-a|^2", type="log", row=1, col=2)
    fig = _layout(fig, "Why dL/dz* is required for real losses")

    final_losses = {mode: float(history[mode]["loss"][-1]) for mode in history}
    summary = (
        f"Final losses -- Wirtinger: {final_losses['wirtinger_conj']:.3e}, "
        f"Split-real: {final_losses['real_split']:.3e}, "
        f"Wrong dL/dz: {final_losses['wrong_dz']:.3e}"
    )
    stats = {
        "history": history,
        "final_losses": final_losses,
    }
    return fig, summary, stats


def plot_iq_map_geometry(
    seed: int = 4,
    n_symbols: int = 256,
    phase_deg: float = 30.0,
    delta_deg: float = 45.0,
) -> tuple[go.Figure, str, dict]:
    """Show rotation-commutation visually for complex vs unconstrained real maps."""
    burst, _ = sample_constellation_burst("qpsk", n_symbols, phase_deg=phase_deg, snr_db=None, seed=seed)

    w_complex = 0.6 + 0.9j
    a_real = np.array([[1.1, 0.45], [-0.2, 0.8]], dtype=np.float64)
    a_complex = complex_map_matrix(w_complex)

    x = burst.astype(np.complex64)
    x_rot = x * np.exp(1j * np.deg2rad(delta_deg))

    def apply(a: np.ndarray, z: np.ndarray) -> np.ndarray:
        iq = np.stack([z.real, z.imag], axis=1)
        out = iq @ a.T
        return out[:, 0] + 1j * out[:, 1]

    complex_rot_then_map = w_complex * x_rot
    complex_map_then_rot = (w_complex * x) * np.exp(1j * np.deg2rad(delta_deg))
    real_rot_then_map = apply(a_real, x_rot)
    real_map_then_rot = apply(a_real, x) * np.exp(1j * np.deg2rad(delta_deg))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Input burst: x and R(x), R = {delta_deg:.0f} deg",
            "Complex map: w(Rx) vs R(wx) (should overlap)",
            "Real 2x2 map: A(Rx) vs R(Ax) (often differs)",
        ),
        horizontal_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(
            x=x.real,
            y=x.imag,
            mode="markers",
            name="x",
            marker={"size": 4, "color": "#1f77b4", "opacity": 0.55},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_rot.real,
            y=x_rot.imag,
            mode="markers",
            name="R(x)",
            marker={"size": 4, "symbol": "x", "color": "#ff7f0e", "opacity": 0.7},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=complex_rot_then_map.real,
            y=complex_rot_then_map.imag,
            mode="markers",
            name="w(Rx)",
            marker={"size": 4, "color": "#2ca02c", "opacity": 0.55},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=complex_map_then_rot.real,
            y=complex_map_then_rot.imag,
            mode="markers",
            name="R(wx)",
            marker={"size": 4, "symbol": "x", "color": "#9467bd", "opacity": 0.75},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=real_rot_then_map.real,
            y=real_rot_then_map.imag,
            mode="markers",
            name="A(Rx)",
            marker={"size": 4, "color": "#d62728", "opacity": 0.55},
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=real_map_then_rot.real,
            y=real_map_then_rot.imag,
            mode="markers",
            name="R(Ax)",
            marker={"size": 4, "symbol": "x", "color": "#8c564b", "opacity": 0.75},
        ),
        row=1,
        col=3,
    )

    all_sets = [x, x_rot, complex_rot_then_map, complex_map_then_rot, real_rot_then_map, real_map_then_rot]
    lim = max(2.0, float(max(np.max(np.abs(z)) for z in all_sets)) * 1.15)
    for c in (1, 2, 3):
        fig.update_xaxes(range=[-lim, lim], title_text="I", row=1, col=c)
        fig.update_yaxes(range=[-lim, lim], title_text="Q", row=1, col=c, scaleanchor=f"x{c}", scaleratio=1)
        fig.add_hline(y=0, line={"color": "#dddddd", "width": 1}, row=1, col=c)
        fig.add_vline(x=0, line={"color": "#dddddd", "width": 1}, row=1, col=c)

    fig.update_layout(height=420)
    fig = _layout(fig, "IQ geometry check: does mapping commute with global rotation?")

    complex_comm_err = rotation_commutation_error_complex(w_complex, x, delta_deg)
    real_comm_err = rotation_commutation_error_real(a_real, x, delta_deg)
    structured_real_err = rotation_commutation_error_real(a_complex, x, delta_deg)
    summary = (
        f"Commutation error mean ||f(Rx)-R(fx)||^2: complex={complex_comm_err:.2e}, "
        f"unconstrained real 2x2={real_comm_err:.2e}, constrained-real(complex form)={structured_real_err:.2e}. "
        "Read the plot left-to-right: panel 2 overlaps (good), panel 3 separates (not rotation-equivariant)."
    )
    stats = {
        "complex_commutation_error": complex_comm_err,
        "real_commutation_error": real_comm_err,
        "structured_real_commutation_error": structured_real_err,
        "real_matrix": a_real,
    }
    return fig, summary, stats


def map_degrees_of_freedom_markdown() -> str:
    """Explain complex scalar maps versus unconstrained real 2x2 maps."""
    w = 0.6 + 0.9j
    a_complex = complex_map_matrix(w)
    return "\n".join(
        [
            "### Degrees of freedom: why the complex map is constrained",
            "",
            "A complex scalar multiply `y = w*x` has 2 real degrees of freedom: `Re(w)` and `Im(w)`. As a real matrix it is always:",
            "",
            "```text",
            "[[ Re(w), -Im(w)],",
            " [ Im(w),  Re(w)]]",
            "```",
            "",
            f"For `w={w:.2f}`, that matrix is approximately:",
            "",
            "```text",
            f"[[{a_complex[0,0]: .2f}, {a_complex[0,1]: .2f}],",
            f" [{a_complex[1,0]: .2f}, {a_complex[1,1]: .2f}]]",
            "```",
            "",
            "A generic real 2x2 map has 4 real degrees of freedom. That extra freedom can learn useful patterns, but it can also shear/warp IQ geometry unless the data teaches it not to.",
        ]
    )


def complex_activation_notes_markdown() -> str:
    """Return a brief caveat about nonlinearities in complex neural nets."""
    return "\n".join(
        [
            "### Complex activations caveat",
            "",
            "Linear complex layers are straightforward, but nonlinearities need care. A usual real activation like ReLU assumes an ordered real line; complex numbers do not have a natural `positive` direction.",
            "",
            "Common complex-network choices include applying nonlinearities to magnitude, phase, or real/imag parts separately. Those choices affect which signal symmetries the model preserves.",
            "",
            "Part 2 handles this in the model design; Part 1 only needs the backprop rule and the IQ geometry intuition.",
        ]
    )


def plot_iq_impairments(
    modulation: str = "qpsk",
    n_symbols: int = 256,
    seed: int = 3,
) -> go.Figure:
    """Show common IQ impairments that motivate rotation-robust models."""
    examples = iq_impairment_examples(modulation=modulation, n_symbols=n_symbols, seed=seed)
    names = list(examples.keys())
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=[name.title() for name in names],
        horizontal_spacing=0.04,
    )

    lim = max(2.0, float(max(np.max(np.abs(z)) for z in examples.values())) * 1.15)
    for col, name in enumerate(names, start=1):
        z = examples[name]
        fig.add_trace(
            go.Scatter(
                x=z.real,
                y=z.imag,
                mode="markers",
                name=name,
                showlegend=False,
                marker={"size": 5, "opacity": 0.65},
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(range=[-lim, lim], title_text="I", row=1, col=col)
        fig.update_yaxes(range=[-lim, lim], title_text="Q", row=1, col=col, scaleanchor=f"x{col}", scaleratio=1)
        fig.add_hline(y=0, line={"color": "#dddddd", "width": 1}, row=1, col=col)
        fig.add_vline(x=0, line={"color": "#dddddd", "width": 1}, row=1, col=col)

    fig.update_layout(height=360)
    return _layout(fig, "Common IQ impairments: same signal family, changed geometry")


def plot_iq_sample_efficiency(
    n_train: int = 24,
    train_phase_deg: float = 15.0,
    n_test: int = 1024,
    noise: float = 0.15,
    seeds: tuple[int, ...] = tuple(range(12)),
    lr: float = 0.3,
    epochs: int = 35,
) -> tuple[go.Figure, str, dict]:
    """Show structured-complex vs unconstrained-real test error over rotation."""
    results = run_iq_rotation_sample_efficiency(
        n_train=n_train,
        n_test=n_test,
        train_phase_deg=train_phase_deg,
        noise=noise,
        seeds=seeds,
        lr=lr,
        epochs=epochs,
    )

    angles = results["angles_deg"]
    complex_mean = results["complex_mean"]
    complex_std = results["complex_std"]
    real_mean = results["real_mean"]
    real_std = results["real_std"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=complex_mean,
            mode="lines",
            name="Complex-structured map (1 complex param)",
            line={"color": "#1f77b4", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=complex_mean + complex_std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=complex_mean - complex_std,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.18)",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=angles,
            y=real_mean,
            mode="lines",
            name="Unconstrained real map (2x2)",
            line={"color": "#d62728", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=real_mean + real_std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=angles,
            y=real_mean - real_std,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.18)",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_vrect(
        x0=-train_phase_deg,
        x1=train_phase_deg,
        fillcolor="rgba(150,150,150,0.16)",
        line_width=0,
        annotation_text="training band",
        annotation_position="top left",
    )
    fig.update_xaxes(title_text="test phase rotation (deg)", tickmode="array", tickvals=np.arange(-180, 181, 45))
    fig.update_yaxes(title_text="MSE on y = w*x regression")
    fig = _layout(fig, "Narrow-band phase training: structured complex map improves sample efficiency")

    summary = (
        f"Train MSE -- complex: {results['complex_train_mean']:.4f}, real: {results['real_train_mean']:.4f}. "
        f"Mean test error gap (real - complex): {results['mean_gap_real_minus_complex']:.4f}. "
        "This gap comes from useful structure constraints, not a different objective (complex: 2 real dof, real 2x2: 4 real dof)."
    )
    return fig, summary, results
