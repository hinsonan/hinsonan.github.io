# Complex vs. Real Neural Networks for IQ Signals

Code for the blog post comparing **complex-valued** neural networks against
**split-real** (I/Q-as-two-channels) baselines on radio-frequency IQ data.

The headline question: *when does a complex-valued network actually beat a
real one?* The answer here is **rotation equivariance**. Complex linear
layers commute with a global phase rotation for free; a real network has to
*learn* that symmetry from data. We pick a task where that property is the
whole game and measure the gap.

## Setup

```bash
conda activate blog-code-examples           # numpy, torch, matplotlib
cd code_examples/complex_vs_real_nn
```

Training uses PyTorch (CUDA if available). CLI plotting uses numpy + matplotlib.
The interactive viewer also needs `gradio`, `plotly`, and `pandas`.

---

## The experiment: rotation-invariant modulation classification

**Task.** Classify which modulation produced a noisy IQ burst — BPSK, QPSK,
8PSK, or 16QAM — under an *unknown carrier phase* (a global rotation of the
constellation). The modulation type is a **rotation-invariant** property: the
answer does not change when you rotate the whole burst.

![modulations](visualizations/modclass_constellations.png)

**The trick that exposes the difference.** We train on a *narrow* rotation band
(θ ∈ ±15°) and test across the *full circle*.

- `ComplexModClassifier` — bias-free `ComplexConv1d` + `modReLU` (each exactly
  rotation-equivariant) finished with a **magnitude pool** (mean/std of `|h|`).
  The whole network is rotation-**invariant by construction**, having seen only
  ±15° in training.
- `ComplexMomentClassifier` — the same equivariant complex feature extractor,
  but with an invariant moment head: magnitude stats plus
  `|mean(unit_phase^2)|`, `|mean(unit_phase^4)|`, and `|mean(unit_phase^8)|`.
  This keeps exact rotation invariance while exposing the cyclic structure that
  separates BPSK/QPSK/8PSK.
- `RealModClassifier` — split-I/Q `Conv1d` + BN + ReLU with stats pooling. It
  can only learn invariance over rotations it was *trained on*.

A third run trains the real net on the **full circle** (`real_full`) to show it
*can* recover invariance — but only by paying for it with augmentation.

### Result

![rotation generalization](visualizations/rotation_generalization.png)

| run | in-band acc (±15°) | full-circle acc | gap |
|-----|-----:|-----:|-----:|
| **complex_narrow** | 0.762 | **0.754** | **+0.007** |
| complex_moment | train this run to update results | train this run to update results | -- |
| real_narrow | 0.970 | 0.631 | +0.339 |
| real_full | 0.730 | 0.728 | +0.002 |

Reading the plot:

- **Complex (blue)** is flat across all 360° — invariant by design, despite
  training on a 24×-narrower band than `real_full`.
- **Real-narrow (red)** hits 0.97 *in-band* (a deceptive number) and collapses
  to 0.45 on unseen rotations. Its peaks land exactly on each constellation's
  rotational symmetry angles (BPSK 180°, QPSK 90°, 8PSK 45°, 16QAM 90°) — see
  `rotation_per_modulation.png` and `confusion_in_vs_ood.png`.
- **Real-full (green)** recovers a flat curve, but the complex net trained on
  ±15° still edges it out over the full circle.

The complex net's invariance is *free and exact*; the real net needs full-circle
data to approximate it.

### Honest caveat

The plain magnitude pool that buys exact invariance also discards the higher-order
phase moment (`E[z⁴]`) that separates the two **constant-modulus** alphabets,
so the complex net confuses QPSK↔8PSK (both unit-circle). BPSK and 16QAM are
near-perfect. This is a genuine representation/invariance tradeoff, discussed in
the post. The `complex_moment` run tests that richer invariant directly.

### Reproduce

```bash
python modclass_cli.py viz       # constellation + rotation-nuisance figures
python modclass_cli.py train     # trains complex_moment, complex_narrow, real_narrow, real_full
python modclass_cli.py eval      # rotation sweep + SNR sweep + confusion plots
python modclass_viewer.py        # Gradio + Plotly interactive data/model viewer
```

Existing checked-in plots may predate `complex_moment`; rerun `train` and `eval`
to regenerate figures with the new model included.

Artifacts: checkpoints/metrics in `trained_modclass/<run>/`, results JSON in
`results/rotation_sweep.json`, figures in `visualizations/`.

---

## Files

| File | Purpose |
|------|---------|
| `modclass_core.py` | Core reusable pieces: `ModClassConfig`, data generation, complex layers, and both model definitions/factories. |
| `modclass_cli.py` | Unified CLI with subcommands: `train`, `eval`, and `viz`. |
| `modclass_viewer.py` | Interactive Gradio/Plotly viewer for bursts, rotations, and checkpoint predictions. |

## Key figures (`visualizations/`)

| File | What it shows |
|------|---------------|
| `modclass_constellations.png` | The four alphabets: ideal vs noisy/rotated received bursts. |
| `modclass_rotation_nuisance.png` | One label, different I/Q pattern per carrier phase. |
| `rotation_generalization.png` | **The money plot** — accuracy vs unseen rotation for available trained runs. |
| `rotation_per_modulation.png` | Real-net dips trace each constellation's rotational symmetry. |
| `snr_sweep_modclass.png` | Full-circle accuracy vs SNR. |
| `confusion_in_vs_ood.png` | Confusion matrices, in-band vs out-of-distribution rotation. |
