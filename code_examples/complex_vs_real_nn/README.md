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

Training uses PyTorch (CUDA if available) and TorchSig for IQ burst generation,
carrier phase rotation, and AWGN. CLI plotting uses numpy + matplotlib.
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

From the repository root:

```bash
conda activate blog-code-examples
cd code_examples/complex_vs_real_nn
```

The experiment is driven by two entry points:

- `python modclass_cli.py ...` — train models, evaluate rotation generalization,
  and generate diagnostics plots.
- `python modclass_viewer.py` — launch a local Gradio/Plotly app for
  interactive inspection of burst samples and model predictions.

#### 1) `viz` — generate the basic figures

What it does:
- Writes `modclass_constellations.png` (ideal constellation vs noisy/rotated
  bursts)
- Writes `modclass_rotation_nuisance.png` (same label, different I/Q pattern at
  different carrier phases)

Run:

```bash
python modclass_cli.py viz
```

Expected output:

```text
saved visualizations/modclass_constellations.png
saved visualizations/modclass_rotation_nuisance.png
```

#### 2) `train` — train the models

What it does:
- Trains the four runs by default: `complex_moment`, `complex_narrow`,
  `real_narrow`, and `real_full`
- Saves checkpoints and metrics under `trained_modclass/<run>/`
- Plots training curves in each run directory

Run:

```bash
python modclass_cli.py train
```

Optional example:

```bash
python modclass_cli.py train --runs complex_narrow real_narrow --epochs 10
```

Expected output:

```text
Device: cpu  | classes: ['bpsk', 'qpsk', '8psk', '16qam']  | SNR: 10 dB
Train/Val sizes: 12000/4000  | epochs: 25

Run: complex_moment  (model=complex_moment, train theta in +/-15 deg)
  params: ...
  epoch ...
  best full-circle acc: ...
...
Summary
complex_moment   in-dist=...  full-circle=...  gap=...
```

Artifacts:
- `trained_modclass/<run>/best_model.pt`
- `trained_modclass/<run>/metrics.json`
- `trained_modclass/<run>/training_curves.png`
- `trained_modclass/<run>/train.log`

#### 3) `eval` — evaluate and make the paper-style figures

What it does:
- Loads the checkpoints from `trained_modclass/`
- Sweeps accuracy across rotations from `-180°` to `+180°`
- Sweeps accuracy across SNRs
- Writes confusion matrices for in-band vs out-of-distribution rotations
- Saves a JSON summary and the figures used in the post

Run:

```bash
python modclass_cli.py eval
```

Expected output:

```text
Device: cpu  classes=['bpsk', 'qpsk', '8psk', '16qam']
  loaded ...
  rotation sweep done: ...
...
saved results/rotation_sweep.json
saved visualizations/rotation_generalization.png
saved visualizations/rotation_per_modulation.png
saved visualizations/snr_sweep_modclass.png
saved visualizations/confusion_in_vs_ood.png
```

Artifacts:
- `results/rotation_sweep.json`
- `visualizations/rotation_generalization.png`
- `visualizations/rotation_per_modulation.png`
- `visualizations/snr_sweep_modclass.png`
- `visualizations/confusion_in_vs_ood.png`

#### 4) `modclass_viewer.py` — interactive browser UI

What it does:
- Launches a local Gradio app that lets you inspect:
  - synthetic IQ bursts
  - rotated/noisy samples
  - prediction probabilities for each trained checkpoint

Run:

```bash
python modclass_viewer.py
```

Expected output:

```text
Running on local URL: http://127.0.0.1:7860
```

Open the printed URL in your browser. If no checkpoints are present yet, the
app will show a warning until you run `python modclass_cli.py train` first.

Existing checked-in plots may predate `complex_moment`; rerun `train` and `eval`
to regenerate figures with the new model included.

---

## Files

| File | Purpose |
|------|---------|
| `modclass_core.py` | Core reusable pieces: `ModClassConfig`, TorchSig-backed data generation, complex layers, and both model definitions/factories. |
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
