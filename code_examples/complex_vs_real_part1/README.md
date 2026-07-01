# Complex vs. Real Neural Networks - Part 1

Part 1 is notebook-first. The article visuals live in a notebook, while reusable
logic lives in Python modules that the notebook imports.

No PyTorch is used here. Everything is numpy plus Plotly.

## Setup

```bash
conda activate blog-code-examples
cd code_examples/complex_vs_real_part1
```

## Quick start

Open and run the notebook:

```bash
jupyter notebook complex_basics.ipynb
```

Run numeric checks (optional but recommended):

```bash
python validation_checks.py
```

## Notebook sections

1. Imaginary numbers as rotation (`i*z`, `exp(i*theta) * z`)
2. Real partial derivatives vs Wirtinger derivatives
3. Why ordinary complex derivatives are not enough for real ML losses
4. Why complex backprop uses the conjugate Wirtinger derivative
5. Complex map geometry, degrees of freedom, and activation caveats
6. Complex sinusoids, IQ trajectories, and common IQ impairments
7. Constellation data preview (BPSK, QPSK, 8PSK, 16QAM) for Part 2

## Files

| File | Purpose |
|------|---------|
| `complex_basics.ipynb` | Main article notebook with visuals and narrative. |
| `complex_core.py` | Core math: rotation, Wirtinger helpers, backprop toy model, IQ geometry checks, impairments, and signal/constellation generators. |
| `complex_plots.py` | Plotly figure builders and markdown formatters used by the notebook. |
| `validation_checks.py` | Finite-difference and training sanity checks for the core math. |

## Bridge to Part 2

Part 2 (`../complex_vs_real_nn/`) uses the same signal framing and constellation
families to compare real vs complex neural networks under phase rotation.
