"""Core math, Wirtinger calculus, and signal helpers for Part 1.

This module is notebook-friendly and depends only on numpy.
"""

from __future__ import annotations

import numpy as np


def magnitude(z: np.ndarray) -> np.ndarray:
    """Return |z|."""
    return np.abs(z)


def phase(z: np.ndarray, deg: bool = False) -> np.ndarray:
    """Return phase angle of z in radians (or degrees)."""
    theta = np.angle(z)
    return np.rad2deg(theta) if deg else theta


def mul_by_i(z: np.ndarray) -> np.ndarray:
    """Multiply by i, equivalent to +90 degree rotation."""
    return 1j * z


def rotate(z: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate complex values by theta_deg."""
    return z * np.exp(1j * np.deg2rad(theta_deg))


def from_dxdy(df_dx: complex, df_dy: complex) -> complex:
    """Convert real partials to df/dz."""
    return 0.5 * (df_dx - 1j * df_dy)


def conj_from_dxdy(df_dx: complex, df_dy: complex) -> complex:
    """Convert real partials to df/dz*."""
    return 0.5 * (df_dx + 1j * df_dy)


def to_wirtinger(*args) -> tuple[complex, complex]:
    """Return (dL/dz, dL/dz*) from real partials.

    Supports both call styles:
      - to_wirtinger(dL_du, dL_dv)
      - to_wirtinger(loss, dL_du, dL_dv)
    """
    if len(args) == 2:
        dL_du, dL_dv = args
    elif len(args) == 3:
        _, dL_du, dL_dv = args
    else:
        raise ValueError("to_wirtinger expects 2 or 3 arguments")
    return from_dxdy(dL_du, dL_dv), conj_from_dxdy(dL_du, dL_dv)


def finite_diff_real_imag(loss_fn, z: complex, eps: float = 1e-5) -> tuple[float, float]:
    """Numerically approximate dL/du and dL/dv at z=u+iv."""
    u, v = z.real, z.imag
    l_c = loss_fn(u + 1j * v)
    l_u = loss_fn((u + eps) + 1j * v)
    l_v = loss_fn(u + 1j * (v + eps))
    return (l_u - l_c) / eps, (l_v - l_c) / eps


def analytic_example(name: str, z: complex, a: complex = 1.0 + 0.0j) -> dict[str, complex]:
    """Return f(z), df/dz, df/dz* for curated examples."""
    b = 0.5 + 0.5j
    if name == "z":
        return {"f": z, "df_dz": 1.0 + 0.0j, "df_dz_conj": 0.0j}
    if name == "z_conj":
        return {"f": np.conj(z), "df_dz": 0.0j, "df_dz_conj": 1.0 + 0.0j}
    if name == "abs2":
        return {"f": np.abs(z) ** 2, "df_dz": np.conj(z), "df_dz_conj": z}
    if name == "az_plus_b_abs2":
        w = a * z + b
        return {"f": np.abs(w) ** 2, "df_dz": a * np.conj(w), "df_dz_conj": np.conj(a) * w}
    if name == "real_z2":
        return {"f": (z ** 2).real, "df_dz": z, "df_dz_conj": np.conj(z)}
    raise ValueError(f"unknown example '{name}'")


def cauchy_riemann_residual(name: str, z: complex, eps: float = 1e-5) -> dict[str, float]:
    """Numerically check Cauchy-Riemann residuals for complex-valued examples."""
    if name == "z2":
        fn = lambda zz: zz**2
    elif name == "conj_z":
        fn = np.conj
    elif name == "abs2_as_complex":
        fn = lambda zz: np.abs(zz) ** 2 + 0.0j
    else:
        raise ValueError(f"unknown Cauchy-Riemann example '{name}'")

    x, y = float(z.real), float(z.imag)
    f0 = fn(x + 1j * y)
    fx = fn((x + eps) + 1j * y)
    fy = fn(x + 1j * (y + eps))

    du_dx = float((fx.real - f0.real) / eps)
    dv_dx = float((fx.imag - f0.imag) / eps)
    du_dy = float((fy.real - f0.real) / eps)
    dv_dy = float((fy.imag - f0.imag) / eps)

    return {
        "du_dx": du_dx,
        "du_dy": du_dy,
        "dv_dx": dv_dx,
        "dv_dy": dv_dy,
        "residual_1": du_dx - dv_dy,
        "residual_2": du_dy + dv_dx,
    }


def iq_impairment_examples(
    modulation: str = "qpsk",
    n_symbols: int = 256,
    seed: int = 3,
    phase_deg: float = 45.0,
    amplitude: float = 1.35,
    snr_db: float = 12.0,
    freq_offset_cycles: float = 0.18,
) -> dict[str, np.ndarray]:
    """Return simple IQ impairment examples for constellation intuition."""
    clean, _ = sample_constellation_burst(modulation, n_symbols, phase_deg=0.0, snr_db=None, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    n = np.arange(n_symbols, dtype=np.float64)
    phase_ramp = np.exp(1j * 2.0 * np.pi * freq_offset_cycles * n / max(1, n_symbols - 1))

    return {
        "clean": clean,
        "phase rotation": clean * np.exp(1j * np.deg2rad(phase_deg)),
        "amplitude scale": amplitude * clean,
        "noise": add_awgn_complex(clean, snr_db, rng),
        "frequency offset": clean * phase_ramp,
    }


def wirtinger_steps(name: str, z: complex) -> dict[str, complex | float | str]:
    """Return intermediate real-partial and Wirtinger derivative steps."""
    x = float(z.real)
    y = float(z.imag)

    if name == "abs2":
        f_expr = "|z|^2 = x^2 + y^2"
        f_val = x**2 + y**2
        df_dx = 2 * x
        df_dy = 2 * y
    elif name == "az_plus_b_abs2":
        f_expr = "|z + (0.5 + 0.5i)|^2 = (x+0.5)^2 + (y+0.5)^2"
        f_val = (x + 0.5) ** 2 + (y + 0.5) ** 2
        df_dx = 2 * (x + 0.5)
        df_dy = 2 * (y + 0.5)
    elif name == "real_z2":
        f_expr = "Re(z^2) = x^2 - y^2"
        f_val = x**2 - y**2
        df_dx = 2 * x
        df_dy = -2 * y
    else:
        raise ValueError(f"step-by-step not implemented for '{name}'")

    df_dz = from_dxdy(df_dx, df_dy)
    df_dz_conj = conj_from_dxdy(df_dx, df_dy)

    return {
        "z": z,
        "x": x,
        "y": y,
        "f_expr": f_expr,
        "f_val": f_val,
        "df_dx": df_dx,
        "df_dy": df_dy,
        "df_dz": df_dz,
        "df_dz_conj": df_dz_conj,
    }


def complex_linear_mse_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: complex,
    b: complex,
) -> tuple[complex, complex, np.ndarray]:
    """Analytic Wirtinger gradients for y_hat = w*x + b with MSE loss."""
    y_hat = w * x + b
    residual = y_hat - y
    dL_dw_conj = np.mean(residual * np.conj(x))
    dL_db_conj = np.mean(residual)
    return dL_dw_conj, dL_db_conj, y_hat


def train_complex_linear(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 50,
    w_init: complex | None = None,
    b_init: complex | None = None,
    seed: int = 7,
) -> dict[str, list[complex] | list[float]]:
    """Run manual gradient descent on a complex linear model."""
    rng = np.random.default_rng(seed)
    w = w_init if w_init is not None else (rng.standard_normal() + 1j * rng.standard_normal())
    b = b_init if b_init is not None else (rng.standard_normal() + 1j * rng.standard_normal())

    history: dict[str, list[complex] | list[float]] = {
        "loss": [],
        "w": [],
        "b": [],
        "grad_w": [],
        "grad_b": [],
    }
    for _ in range(epochs):
        gw, gb, y_hat = complex_linear_mse_gradients(x, y, w, b)
        loss = np.mean(np.abs(y_hat - y) ** 2)
        history["loss"].append(float(loss))
        history["w"].append(w)
        history["b"].append(b)
        history["grad_w"].append(gw)
        history["grad_b"].append(gb)
        w = w - lr * gw
        b = b - lr * gb
    return history


def toy_quadratic_loss(w: complex, target: complex) -> float:
    """Return L(w) = |w - target|^2."""
    return float(np.abs(w - target) ** 2)


def toy_quadratic_gradients(w: complex, target: complex) -> tuple[complex, complex, float, float]:
    """Return (dL/dw, dL/dw*, dL/du, dL/dv) for L(w)=|w-target|^2."""
    diff = w - target
    dL_dw = np.conj(diff)
    dL_dw_conj = diff
    dL_du = float(2.0 * diff.real)
    dL_dv = float(2.0 * diff.imag)
    return dL_dw, dL_dw_conj, dL_du, dL_dv


def simulate_descent_methods(
    target: complex = 1.0 + 2.0j,
    w_init: complex = -2.0 + 1.5j,
    lr: float = 0.25,
    epochs: int = 20,
) -> dict[str, dict[str, list[complex] | list[float]]]:
    """Compare descent using Wirtinger-conj, real-split, and wrong dL/dz updates."""
    modes = ("wirtinger_conj", "real_split", "wrong_dz")
    state = {mode: w_init for mode in modes}
    history: dict[str, dict[str, list[complex] | list[float]]] = {
        mode: {"w": [w_init], "loss": [toy_quadratic_loss(w_init, target)]} for mode in modes
    }

    for _ in range(epochs):
        for mode in modes:
            w = state[mode]
            dL_dw, dL_dw_conj, dL_du, dL_dv = toy_quadratic_gradients(w, target)

            if mode == "wirtinger_conj":
                w_next = w - lr * dL_dw_conj
            elif mode == "real_split":
                u_next = w.real - 0.5 * lr * dL_du
                v_next = w.imag - 0.5 * lr * dL_dv
                w_next = u_next + 1j * v_next
            else:
                w_next = w - lr * dL_dw

            state[mode] = w_next
            history[mode]["w"].append(w_next)
            history[mode]["loss"].append(toy_quadratic_loss(w_next, target))

    return history


def complex_map_matrix(w: complex) -> np.ndarray:
    """Return the 2x2 real matrix induced by complex multiplication y=w*x."""
    return np.array(
        [
            [w.real, -w.imag],
            [w.imag, w.real],
        ],
        dtype=np.float64,
    )


def apply_real_matrix_map(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x2 real matrix to IQ samples represented as complex numbers."""
    x_iq = np.stack([x.real, x.imag], axis=1)
    y_iq = x_iq @ matrix.T
    return (y_iq[:, 0] + 1j * y_iq[:, 1]).astype(np.complex64)


def rotation_commutation_error_complex(w: complex, x: np.ndarray, delta_deg: float) -> float:
    """Measure ||w*R(x) - R(w*x)||^2 mean for complex multiplication."""
    x_rot = x * np.exp(1j * np.deg2rad(delta_deg))
    y1 = w * x_rot
    y2 = (w * x) * np.exp(1j * np.deg2rad(delta_deg))
    return float(np.mean(np.abs(y1 - y2) ** 2))


def rotation_commutation_error_real(matrix: np.ndarray, x: np.ndarray, delta_deg: float) -> float:
    """Measure ||A*R(x) - R(A*x)||^2 mean for a real 2x2 map."""
    x_rot = x * np.exp(1j * np.deg2rad(delta_deg))
    y1 = apply_real_matrix_map(x_rot, matrix)
    y2 = apply_real_matrix_map(x, matrix) * np.exp(1j * np.deg2rad(delta_deg))
    return float(np.mean(np.abs(y1 - y2) ** 2))


def generate_iq_linear_task(
    n_samples: int,
    phase_low_deg: float,
    phase_high_deg: float,
    noise: float = 0.15,
    seed: int = 0,
    modulation: str = "qpsk",
    w_true: complex = 0.35 + 1.05j,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """Generate x,y pairs for y = w_true*x + noise with controlled phase range."""
    rng = np.random.default_rng(seed)
    pts = CONSTELLATIONS[modulation]
    idx = rng.integers(0, len(pts), size=n_samples)
    symbols = pts[idx]

    theta_deg = rng.uniform(phase_low_deg, phase_high_deg, size=n_samples)
    x = symbols * np.exp(1j * np.deg2rad(theta_deg))
    eps = noise * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    y = w_true * x + eps
    return x.astype(np.complex64), y.astype(np.complex64), theta_deg.astype(np.float32), w_true


def train_complex_scalar_map(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.3,
    epochs: int = 35,
    w_init: complex = 0.0 + 0.0j,
) -> dict[str, list[complex] | list[float]]:
    """Fit y~=w*x with one complex parameter using Wirtinger-conj updates."""
    w = w_init
    history: dict[str, list[complex] | list[float]] = {"w": [w], "loss": []}
    for _ in range(epochs):
        residual = w * x - y
        loss = float(np.mean(np.abs(residual) ** 2))
        grad = np.mean(residual * np.conj(x))
        w = w - lr * grad
        history["loss"].append(loss)
        history["w"].append(w)
    return history


def train_real_matrix_map(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.3,
    epochs: int = 35,
    matrix_init: np.ndarray | None = None,
) -> dict[str, list[np.ndarray] | list[float]]:
    """Fit y_iq~=A*x_iq with an unconstrained 2x2 real matrix.

    Uses the true real gradient for L = mean(||A x_iq - y_iq||^2):
        dL/dA = 2 * (residual^T x_iq) / n
    """
    a = np.array(matrix_init, dtype=np.float64) if matrix_init is not None else np.zeros((2, 2), dtype=np.float64)
    x_iq = np.stack([x.real, x.imag], axis=1)
    y_iq = np.stack([y.real, y.imag], axis=1)
    n = float(x_iq.shape[0])

    history: dict[str, list[np.ndarray] | list[float]] = {"matrix": [a.copy()], "loss": []}
    for _ in range(epochs):
        residual = x_iq @ a.T - y_iq
        loss = float(np.mean(np.sum(residual**2, axis=1)))
        grad = 2.0 * (residual.T @ x_iq) / n
        a = a - lr * grad
        history["loss"].append(loss)
        history["matrix"].append(a.copy())
    return history


def mse_complex_scalar_map(w: complex, x: np.ndarray, y: np.ndarray) -> float:
    """Return MSE for y_hat = w*x."""
    return float(np.mean(np.abs(w * x - y) ** 2))


def mse_real_matrix_map(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Return MSE for y_hat_iq = matrix * x_iq."""
    x_iq = np.stack([x.real, x.imag], axis=1)
    y_iq = np.stack([y.real, y.imag], axis=1)
    residual = x_iq @ matrix.T - y_iq
    return float(np.mean(np.sum(residual**2, axis=1)))


def run_iq_rotation_sample_efficiency(
    n_train: int = 24,
    n_test: int = 1024,
    train_phase_deg: float = 15.0,
    test_angles_deg: np.ndarray | None = None,
    noise: float = 0.15,
    seeds: tuple[int, ...] = tuple(range(12)),
    lr: float = 0.3,
    epochs: int = 35,
) -> dict[str, np.ndarray | float]:
    """Compare complex vs split-real linear maps under narrow-band phase training.

    We use lr_real = 0.5 * lr so per-coordinate update scale is aligned with the
    complex dL/dw* convention used in train_complex_scalar_map.
    """
    angles = np.array(test_angles_deg if test_angles_deg is not None else np.arange(-180.0, 181.0, 15.0), dtype=np.float64)

    complex_err = np.zeros((len(seeds), len(angles)), dtype=np.float64)
    real_err = np.zeros((len(seeds), len(angles)), dtype=np.float64)
    train_complex = np.zeros(len(seeds), dtype=np.float64)
    train_real = np.zeros(len(seeds), dtype=np.float64)

    for si, seed in enumerate(seeds):
        x_train, y_train, _, _ = generate_iq_linear_task(
            n_samples=n_train,
            phase_low_deg=-train_phase_deg,
            phase_high_deg=train_phase_deg,
            noise=noise,
            seed=seed,
        )
        complex_hist = train_complex_scalar_map(x_train, y_train, lr=lr, epochs=epochs)
        real_hist = train_real_matrix_map(x_train, y_train, lr=0.5 * lr, epochs=epochs)

        w_hat = complex_hist["w"][-1]
        a_hat = real_hist["matrix"][-1]
        train_complex[si] = mse_complex_scalar_map(w_hat, x_train, y_train)
        train_real[si] = mse_real_matrix_map(a_hat, x_train, y_train)

        for ai, angle in enumerate(angles):
            x_test, y_test, _, _ = generate_iq_linear_task(
                n_samples=n_test,
                phase_low_deg=angle,
                phase_high_deg=angle,
                noise=noise,
                seed=seed + 10_000 + ai * 37,
            )
            complex_err[si, ai] = mse_complex_scalar_map(w_hat, x_test, y_test)
            real_err[si, ai] = mse_real_matrix_map(a_hat, x_test, y_test)

    return {
        "angles_deg": angles,
        "complex_mean": complex_err.mean(axis=0),
        "complex_std": complex_err.std(axis=0),
        "real_mean": real_err.mean(axis=0),
        "real_std": real_err.std(axis=0),
        "complex_train_mean": float(train_complex.mean()),
        "real_train_mean": float(train_real.mean()),
        "mean_gap_real_minus_complex": float(np.mean(real_err - complex_err)),
    }


def make_complex_regression_data(
    n_samples: int,
    noise: float = 0.1,
    seed: int = 0,
    w_true: complex = 0.8 - 0.5j,
    b_true: complex = 0.2 + 0.1j,
) -> tuple[np.ndarray, np.ndarray, complex, complex]:
    """Generate synthetic data for y = w*x + b with complex noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    y = w_true * x + b_true + noise * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    return x, y, w_true, b_true


def add_awgn_complex(z: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN to z at a given SNR in dB."""
    signal_power = float(np.mean(np.abs(z) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)) * sigma
    return z + noise


def complex_sinusoid(
    n: int,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase_deg: float = 0.0,
    snr_db: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate z[t] = A * exp(i * (2*pi*f*t + phi))."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    phi = np.deg2rad(phase_deg)
    z = amplitude * np.exp(1j * (2.0 * np.pi * frequency * t + phi)).astype(np.complex64)
    if snr_db is not None:
        z = add_awgn_complex(z, snr_db, rng)
    return z, t


def _bpsk() -> np.ndarray:
    pts = np.array([1.0, -1.0], dtype=np.complex64)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _qpsk() -> np.ndarray:
    pts = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex64)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _psk8() -> np.ndarray:
    k = np.arange(8)
    pts = np.exp(1j * 2 * np.pi * k / 8).astype(np.complex64)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _qam16() -> np.ndarray:
    levels = np.array([-3, -1, 1, 3], dtype=np.float32)
    re, im = np.meshgrid(levels, levels)
    pts = (re + 1j * im).astype(np.complex64).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS: dict[str, np.ndarray] = {
    "bpsk": _bpsk(),
    "qpsk": _qpsk(),
    "8psk": _psk8(),
    "16qam": _qam16(),
}


MODULATION_NOTES: dict[str, dict[str, str]] = {
    "bpsk": {
        "name": "BPSK",
        "what": "Binary phase-shift keying with 2 phase states (1 bit/symbol).",
        "uses": "Robust low-rate links like telemetry, satellite control, and deep-space style channels.",
    },
    "qpsk": {
        "name": "QPSK",
        "what": "Quadrature phase-shift keying with 4 phase states (2 bits/symbol).",
        "uses": "Widely used in cellular, satellite, and many digital radio systems.",
    },
    "8psk": {
        "name": "8PSK",
        "what": "Phase-shift keying with 8 phase states (3 bits/symbol).",
        "uses": "Used when higher spectral efficiency is needed, for example in some satellite broadcast links.",
    },
    "16qam": {
        "name": "16QAM",
        "what": "Quadrature amplitude modulation with 16 amplitude-phase points (4 bits/symbol).",
        "uses": "Common in Wi-Fi, LTE/5G, cable modems, and other high-throughput links.",
    },
}


def phase_shift_same_signal(z: np.ndarray, delta_phase_deg: float) -> np.ndarray:
    """Rotate all samples by the same phase offset."""
    return z * np.exp(1j * np.deg2rad(delta_phase_deg))


def sample_constellation_burst(
    name: str,
    n_symbols: int,
    phase_deg: float = 0.0,
    snr_db: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample symbols from a named constellation and apply global phase/noise."""
    rng = np.random.default_rng(seed)
    pts = CONSTELLATIONS[name]
    idx = rng.integers(0, len(pts), size=n_symbols)
    clean = pts[idx]
    burst = (clean * np.exp(1j * np.deg2rad(phase_deg))).astype(np.complex64)
    if snr_db is not None:
        burst = add_awgn_complex(burst, snr_db, rng)
    return burst, clean
