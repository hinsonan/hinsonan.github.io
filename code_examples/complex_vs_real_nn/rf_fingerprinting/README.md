# RF Fingerprinting: Complex vs Real Neural Encoders

This module is a lightweight, notebook-first project for comparing real-valued and
complex-valued neural encoders on RF fingerprinting tasks.

## What this includes

- NPZ data loader for RF captures with expected keys:
  - `iq` (`complex64`, shape `[N, T]`)
  - `device_id` (`int64`, shape `[N]`)
  - optional `session_id` (`int64`, shape `[N]`)
- Synthetic fallback dataset generator when no NPZ file is provided/found.
- SimCLR-style pretraining (`TwoView` dataset + NT-Xent loss).
- Matching embedding dimension for:
  - real encoder (`models_real.py`)
  - complex encoder (`models_complex.py`)
- Linear probe, supervised fine-tuning, evaluation, and open-set helpers.
- Notebook workflow in `rf_fingerprint.ipynb` as the main entry point.

## Install

Minimal dependencies:

```bash
python -m pip install numpy torch matplotlib scikit-learn
```

Optional (for YAML config loading):

```bash
python -m pip install pyyaml
```

## Notebook-first workflow

Open and run:

- `rf_fingerprint.ipynb`

The notebook defaults to a short runtime (`fast` mode):

1. Setup config
2. Load NPZ data (or generate synthetic fallback)
3. Preview augmentations
4. Initialize encoder
5. Short SimCLR pretraining
6. Linear probe
7. Short fine-tune
8. Evaluate + export scorecard CSV

## CLI scripts

From this directory:

```bash
python scripts/run_pretrain.py --config configs/rf_fp_fast.yaml
python scripts/run_probe.py --config configs/rf_fp_fast.yaml --encoder real
python scripts/run_finetune.py --config configs/rf_fp_fast.yaml --encoder real
python scripts/run_eval.py --config configs/rf_fp_fast.yaml --encoder real
```

## Configuration presets

- `configs/rf_fp_base.yaml`: balanced default
- `configs/rf_fp_fast.yaml`: shortest runtime
- `configs/rf_fp_full.yaml`: larger synthetic dataset and more epochs
