# RF Fingerprinting: Complex vs Real CNNs

A small experiment asking one question: can a neural network distinguish two
emitters that transmit the same QPSK waveform but have different hardware?

TorchSig generates the QPSK signals and applies a fixed fingerprint to each
synthetic emitter: IQ imbalance, oscillator phase noise, and amplifier
nonlinearity. Per-capture gain, phase, and noise vary independently so the
network cannot memorize channel conditions.

The notebook compares similarly sized real and complex 1D CNNs using:

- closed-set emitter identification accuracy;
- confusion matrices and embedding plots;
- robustness as test SNR changes;
- one optional unknown-emitter AUROC score.

## Run

```bash
cd code_examples/complex_vs_real_nn/rf_fingerprinting
jupyter notebook rf_fingerprint.ipynb
```

The implementation is intentionally small:

```text
signals.py       TorchSig data generation
models.py        real and complex CNNs
training.py      supervised training and embeddings
evaluation.py    confusion matrix and open-set score
rf_fingerprint.ipynb
```

Dependencies are listed in `code_examples/requirements.txt`, including
`torchsig==2.1.1`.
