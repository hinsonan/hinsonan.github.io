"""Numeric validation checks for complex-core math.

Run:
    python validation_checks.py
"""

import numpy as np

from complex_core import (
    analytic_example,
    complex_linear_mse_gradients,
    complex_map_matrix,
    cauchy_riemann_residual,
    finite_diff_real_imag,
    iq_impairment_examples,
    make_complex_regression_data,
    rotation_commutation_error_complex,
    rotation_commutation_error_real,
    run_iq_rotation_sample_efficiency,
    simulate_descent_methods,
    to_wirtinger,
    train_complex_linear,
)


def check_analytic_vs_finite_diff(name: str, z: complex, tol: float = 1e-4) -> bool:
    vals = analytic_example(name, z)

    if name == "abs2":
        loss_fn = lambda zp: abs(zp) ** 2
    elif name == "az_plus_b_abs2":
        b = 0.5 + 0.5j
        loss_fn = lambda zp: abs(zp + b) ** 2
    elif name == "real_z2":
        loss_fn = lambda zp: (zp**2).real
    else:
        raise ValueError(f"Unsupported check name '{name}'")

    dL_du_fd, dL_dv_fd = finite_diff_real_imag(loss_fn, z)
    dL_dz_fd, dL_dz_conj_fd = to_wirtinger(dL_du_fd, dL_dv_fd)

    err = abs(dL_dz_fd - vals["df_dz"]) + abs(dL_dz_conj_fd - vals["df_dz_conj"])
    ok = err < tol
    print(f"  {name:20s} err={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_complex_linear_gradients(tol: float = 1e-4) -> bool:
    rng = np.random.default_rng(42)
    n = 32
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    w = 0.5 - 0.3j
    b = 0.1 + 0.2j

    gw, gb, _ = complex_linear_mse_gradients(x, y, w, b)

    eps = 1e-5

    def loss_fn(uw, vw, ub, vb):
        wp = uw + 1j * vw
        bp = ub + 1j * vb
        return np.mean(np.abs(wp * x + bp - y) ** 2)

    l0 = loss_fn(w.real, w.imag, b.real, b.imag)
    l_wu = loss_fn(w.real + eps, w.imag, b.real, b.imag)
    l_wv = loss_fn(w.real, w.imag + eps, b.real, b.imag)
    l_bu = loss_fn(w.real, w.imag, b.real + eps, b.imag)
    l_bv = loss_fn(w.real, w.imag, b.real, b.imag + eps)

    _, gw_conj_fd = to_wirtinger((l_wu - l0) / eps, (l_wv - l0) / eps)
    _, gb_conj_fd = to_wirtinger((l_bu - l0) / eps, (l_bv - l0) / eps)

    err_w = abs(gw - gw_conj_fd)
    err_b = abs(gb - gb_conj_fd)
    ok = (err_w < tol) and (err_b < tol)
    print(f"  linear grad w err={err_w:.3e}, b err={err_b:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_training_reduces_loss() -> bool:
    x, y, _, _ = make_complex_regression_data(n_samples=128, noise=0.05, seed=123)
    history = train_complex_linear(x, y, lr=0.1, epochs=30, seed=99)
    initial = history["loss"][0]
    final = history["loss"][-1]
    ok = final < initial
    print(f"  training loss {initial:.4e} -> {final:.4e} {'OK' if ok else 'FAIL'}")
    return ok


def check_descent_equivalence_and_wrong_mode(tol: float = 1e-8) -> bool:
    history = simulate_descent_methods(target=1.0 + 2.0j, w_init=-2.0 + 1.5j, lr=0.25, epochs=20)
    w_wirt = np.array(history["wirtinger_conj"]["w"])
    w_real = np.array(history["real_split"]["w"])
    l_wirt = np.array(history["wirtinger_conj"]["loss"])
    l_wrong = np.array(history["wrong_dz"]["loss"])

    traj_match = float(np.max(np.abs(w_wirt - w_real)))
    wrong_worse = float(l_wrong[-1]) > float(l_wirt[-1])
    ok = (traj_match < tol) and wrong_worse
    print(
        f"  trajectory match={traj_match:.3e}, wrong-final-loss={l_wrong[-1]:.3e}, "
        f"correct-final-loss={l_wirt[-1]:.3e} {'OK' if ok else 'FAIL'}"
    )
    return ok


def check_rotation_commutation() -> bool:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    w = 0.6 + 0.9j
    a_bad = np.array([[1.1, 0.45], [-0.2, 0.8]], dtype=np.float64)
    a_good = complex_map_matrix(w)

    e_complex = rotation_commutation_error_complex(w, x, 45.0)
    e_bad = rotation_commutation_error_real(a_bad, x, 45.0)
    e_good = rotation_commutation_error_real(a_good, x, 45.0)

    ok = (e_complex < 1e-10) and (e_good < 1e-10) and (e_bad > 1e-2)
    print(f"  commutation error complex={e_complex:.3e}, unconstrained-real={e_bad:.3e}, constrained-real={e_good:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_iq_sample_efficiency() -> bool:
    results = run_iq_rotation_sample_efficiency(
        n_train=24,
        n_test=512,
        train_phase_deg=15.0,
        noise=0.15,
        seeds=tuple(range(8)),
        lr=0.3,
        epochs=35,
    )
    gap = float(results["mean_gap_real_minus_complex"])
    ok = gap > 0.0
    print(f"  mean test-error gap (real-complex) = {gap:.4e} {'OK' if ok else 'FAIL'}")
    return ok


def check_optimizer_step_equivalence() -> bool:
    rng = np.random.default_rng(13)
    w = rng.standard_normal() + 1j * rng.standard_normal()
    grad_conj = rng.standard_normal() + 1j * rng.standard_normal()
    lr = 0.17

    w_complex = w - lr * grad_conj
    wr = np.array([w.real, w.imag], dtype=np.float64)
    gr = np.array([grad_conj.real, grad_conj.imag], dtype=np.float64)
    wr_next = wr - lr * gr
    w_real = wr_next[0] + 1j * wr_next[1]

    err = abs(w_complex - w_real)
    ok = err < 1e-12
    print(f"  complex-step vs split-real-step mismatch={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_cauchy_riemann_examples() -> bool:
    z = 1.2 - 0.7j
    hol = cauchy_riemann_residual("z2", z)
    non_hol = cauchy_riemann_residual("abs2_as_complex", z)
    hol_res = abs(hol["residual_1"]) + abs(hol["residual_2"])
    non_hol_res = abs(non_hol["residual_1"]) + abs(non_hol["residual_2"])
    ok = hol_res < 1e-3 and non_hol_res > 1e-2
    print(f"  CR residual z^2={hol_res:.3e}, |z|^2={non_hol_res:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def check_iq_impairments() -> bool:
    examples = iq_impairment_examples(n_symbols=128, seed=4)
    keys_ok = set(examples) == {"clean", "phase rotation", "amplitude scale", "noise", "frequency offset"}
    shapes_ok = all(v.shape == examples["clean"].shape for v in examples.values())
    finite_ok = all(np.all(np.isfinite(v.real)) and np.all(np.isfinite(v.imag)) for v in examples.values())
    ok = keys_ok and shapes_ok and finite_ok
    print(f"  impairment examples keys={len(examples)}, shape={examples['clean'].shape} {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Part 1 numeric checks")
    print("=" * 60)

    ok = True
    print("\n1) Wirtinger derivatives vs finite differences")
    for name in ["abs2", "az_plus_b_abs2", "real_z2"]:
        ok &= check_analytic_vs_finite_diff(name, 1.2 - 0.7j)

    print("\n2) Complex linear regression gradients")
    ok &= check_complex_linear_gradients()

    print("\n3) Manual training reduces loss")
    ok &= check_training_reduces_loss()

    print("\n4) Wirtinger vs split-real and wrong dL/dz")
    ok &= check_descent_equivalence_and_wrong_mode()

    print("\n5) IQ geometry commutation check")
    ok &= check_rotation_commutation()

    print("\n6) IQ narrow-band sample-efficiency check")
    ok &= check_iq_sample_efficiency()

    print("\n7) Optimizer step equivalence (complex vs split-real)")
    ok &= check_optimizer_step_equivalence()

    print("\n8) Holomorphic vs non-holomorphic examples")
    ok &= check_cauchy_riemann_examples()

    print("\n9) IQ impairment examples")
    ok &= check_iq_impairments()

    print("\n" + "=" * 60)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
