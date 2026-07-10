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

Edit ``MODE`` and rerun the notebook cells to switch between compute presets.

### Impairment controls

The notebook also exposes independent source and capture controls:

```python
IMPAIRMENT_PROFILE = "controlled"
IMPAIRMENT_ABLATION = None       # e.g. "no_cfo" or "no_channel"
EVALUATE_ENCODERS = ("real", "complex")
COMPLEX_POOLING = "stats"       # "avg" is magnitude-only; "stats" keeps phase statistics
KNOWN_DEVICE_COUNT = 5
UNKNOWN_DEVICE_COUNT = 5
UNKNOWN_CALIBRATION_DEVICE_COUNT = 2
```

Available profiles are ``oracle``, ``device_full``, ``controlled``,
``receiver_only``, ``stress_channel``, ``stress_waveform``, ``default``, and
``full``. The recommended progression is ``oracle`` to verify the negative
control, ``device_full`` to establish a device-only upper bound, ``controlled``
as the learnable baseline, then the stress profiles before using ``full``.

``controlled`` intentionally disables session multipath, the second channel
stage, and waveform variation. Those effects currently overwhelm the device
signature in session-held-out tests, so they should be introduced one at a
time rather than treated as the default training condition.

Available ablations include ``no_iq``, ``no_cfo``, ``no_phase_noise``,
``no_pa``, ``no_memory``, ``no_dc``, ``no_session_channel``, ``no_channel``,
``no_receiver``, ``no_waveform_variation``, ``no_awgn``, and
``no_quantization``.

The resolved settings are printed before generation. For custom ranges,
modify ``cfg.device_impairments`` or ``cfg.nuisance_impairments`` after
``load_config`` and call ``cfg.validate_impairments()`` before generating data.

The notebook defaults to five enrolled devices and five separate unknown
devices. Unknown devices are excluded from training and validation, while all
generated devices share the same configured signal-generation process.
Two unknown devices are reserved for development-time cosine-threshold
calibration; the remaining unknown devices are held out for final evaluation.

The complex encoder's ``avg`` pooling discards activation phase and retains
only magnitude. ``stats`` is the recommended setting for fingerprinting
because it retains real/imaginary statistics and circular phase moments.

The notebook also compares open-set scores without retraining: maximum
softmax, top-two logit margin, negative entropy, log-sum-exp score, and
maximum cosine similarity to known-device embedding prototypes. Higher scores
always indicate greater known-device confidence; each score's AUROC is saved
in the scorecard with an ``open_set_<score>_auroc`` column.

Prototype cosine is the primary open-set decision score. Known-device
prototypes are computed from training embeddings, a threshold is calibrated
using validation known samples and development unknown devices, and final
open-set AUROC/OSCR use prototype confidence. Max-softmax and other logit
scores remain available as comparison columns.

## Dependencies

```bash
python -m pip install numpy torch matplotlib scikit-learn
```
