# Complex vs Real NN Workspace

This workspace now contains two related modules:

- `modulation_classification/` — existing rotation-invariant modulation
  classification project comparing complex-valued and real-valued models.
- `rf_fingerprinting/` — new SimCLR-style RF fingerprinting project with
  real vs complex encoder comparisons.

## Quick Start

```bash
conda activate blog-code-examples
python -m pip install -r code_examples/requirements.txt
```

### Modulation classification

```bash
cd code_examples/complex_vs_real_nn/modulation_classification
jupyter notebook complex_vs_real_nn.ipynb
```

### RF fingerprinting

```bash
cd code_examples/complex_vs_real_nn/rf_fingerprinting
jupyter notebook rf_fingerprint.ipynb
```
