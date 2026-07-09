# RF Fingerprinting: Complex vs Real Encoders

Notebook-first comparison of real-valued and complex-valued neural encoders for RF fingerprinting.

## Structure

```
rf_fingerprinting/
├── rf_fingerprint.ipynb   # main entry point
├── config.py              # config presets and small helpers
├── data.py                # data loading + synthetic generator
├── datasets.py            # datasets, splits, augmentations
├── models.py              # real/complex encoders and heads
├── training.py            # SimCLR, fine-tune, probe, eval
└── visualize.py           # plotting helpers
```

No nested package, no scripts, no YAML configs. All interaction happens through the notebook.

## Quick start

```bash
cd code_examples/complex_vs_real_nn/rf_fingerprinting
jupyter notebook rf_fingerprint.ipynb
```

The notebook has a ``MODE`` cell at the top:

- ``MODE = "fast"`` — small synthetic dataset, short training (smoke test / iteration)
- ``MODE = "base"`` — balanced default
- ``MODE = "full"`` — larger dataset and longer training

Edit ``MODE`` and rerun the notebook cells to switch between profiles.

## Dependencies

```bash
python -m pip install numpy torch matplotlib scikit-learn
```
