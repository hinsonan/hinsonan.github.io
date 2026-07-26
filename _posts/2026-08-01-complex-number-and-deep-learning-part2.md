---
layout: post
title: "Complex Numbers and Deep Learning Part 2"
date: 2026-08-01
categories: ML
---

It's time to process some signals and compare "real" neural networks and "complex" neural networks. First off let's explain what in the world we are going to attempt. [Part 1]({% post_url 2026-07-01-complex-number-and-deep-learning-part1 %}) was all about setting up the foundation for how complex numbers work. We learned how to use Wirtinger derivatives to backprop and train models. We learned that complex layers should be able to capture rotations and natural relationships a bit more efficiently than real numbers.

The goal now is to test some complex and real models on the same problem and see what happens.

## Modulation Classification

Our task is to take in received bursts of IQ data. Once we receive these bursts we want to classify them into their modulation.

The modulations we are looking at is BPSK, QPSK, 8PSK, and 16QAM. These are the four class labels we want to try and classify. Signals do not come in clean. In the real world signals will contain a lot of noise. Every signal is rotated and additive white gaussian noise is applied to these signals. These types of rotations are what you would see in the real world and in many cases you would have to deal with many other outside influences.

- **BPSK (Binary Phase Shift Keying):** Uses two phase states to represent one bit per symbol. It is a robust choice for weak or noisy links, including satellite and GPS signals.
- **QPSK (Quadrature Phase Shift Keying):** Uses four phase states to represent two bits per symbol. It is common in satellite communication, cellular systems, and Wi-Fi control channels.
- **8PSK (8-Phase Shift Keying):** Uses eight phase states to represent three bits per symbol. It increases data rate over QPSK but needs a cleaner channel because its points are closer together.
- **16QAM (16-Quadrature Amplitude Modulation):** Uses both amplitude and phase to represent four bits per symbol. It is used in higher-throughput Wi-Fi, LTE, 5G, and cable-modem links when signal conditions are good.

If you remember from part 1 these rotations are why we want to test some complex layers. We should be able to learn these natural rotational relationships easier with complex layers.

This interactive plot shows the types of signal data we are going to be looking at. The left shows the constellation plot and the right shows the IQ data broken out into in phase and quadrature.

<div id="modclass-signal-demo" style="max-width:900px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;">
  <canvas id="modclass-signal-canvas" width="860" height="400" style="width:100%;height:auto;display:block;border-radius:6px;background:#10161d;"></canvas>
  <div style="margin-top:0.7rem;display:flex;gap:0.7rem;flex-wrap:wrap;align-items:center;font-size:0.85rem;">
    <label style="display:flex;gap:0.35rem;align-items:center;">modulation
      <select id="modclass-modulation" style="padding:0.2rem 0.35rem;border:1px solid #555;border-radius:4px;background:#1b242d;color:#eee;">
        <option value="bpsk">BPSK</option>
        <option value="qpsk" selected>QPSK</option>
        <option value="8psk">8PSK</option>
        <option value="16qam">16QAM</option>
      </select>
    </label>
    <label style="display:flex;gap:0.35rem;align-items:center;">carrier phase
      <input id="modclass-phase" type="range" min="-180" max="180" step="5" value="35" style="width:105px;">
      <span id="modclass-phase-value" style="min-width:3.5em;">35°</span>
    </label>
    <label style="display:flex;gap:0.35rem;align-items:center;">SNR
      <input id="modclass-snr" type="range" min="0" max="30" step="1" value="10" style="width:90px;">
      <span id="modclass-snr-value" style="min-width:3.5em;">10 dB</span>
    </label>
    <button id="modclass-regenerate" type="button" style="padding:0.25rem 0.6rem;border:1px solid #4db0ff;border-radius:4px;background:transparent;color:#4db0ff;cursor:pointer;">new burst</button>
  </div>
  <p style="margin:0.65rem 0 0;color:#aaa;font-size:0.82rem;">128 symbols; a single phase offset applies to the whole burst. The model must classify the alphabet, not the rotation.</p>
</div>

<script>
(() => {
  const canvas = document.getElementById('modclass-signal-canvas');
  const modulation = document.getElementById('modclass-modulation');
  const phase = document.getElementById('modclass-phase');
  const snr = document.getElementById('modclass-snr');
  const regenerate = document.getElementById('modclass-regenerate');
  if (!canvas || !modulation || !phase || !snr || !regenerate) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const colors = { i: '#4db0ff', q: '#ffb74d', ideal: 'rgba(160,190,210,0.45)' };
  const alphabets = {
    bpsk: [{ re: -1, im: 0 }, { re: 1, im: 0 }],
    qpsk: [-1, 1].flatMap(re => [-1, 1].map(im => ({ re: re / Math.SQRT2, im: im / Math.SQRT2 }))),
    '8psk': Array.from({ length: 8 }, (_, k) => ({ re: Math.cos(2 * Math.PI * k / 8), im: Math.sin(2 * Math.PI * k / 8) })),
    '16qam': [-3, -1, 1, 3].flatMap(re => [-3, -1, 1, 3].map(im => ({ re: re / Math.sqrt(10), im: im / Math.sqrt(10) })))
  };
  let burst = [];

  function gaussian() {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function newBurst() {
    const points = alphabets[modulation.value];
    burst = Array.from({ length: 128 }, () => ({ ...points[Math.floor(Math.random() * points.length)] }));
  }

  function received() {
    const theta = Number(phase.value) * Math.PI / 180;
    const signalPower = burst.reduce((sum, z) => sum + z.re * z.re + z.im * z.im, 0) / burst.length;
    const noiseSigma = Math.sqrt(signalPower / (2 * Math.pow(10, Number(snr.value) / 10)));
    return burst.map(z => ({
      re: z.re * Math.cos(theta) - z.im * Math.sin(theta) + noiseSigma * gaussian(),
      im: z.re * Math.sin(theta) + z.im * Math.cos(theta) + noiseSigma * gaussian()
    }));
  }

  function text(value, x, y, color = '#aaa', size = 12) {
    ctx.fillStyle = color;
    ctx.font = `${size}px sans-serif`;
    ctx.fillText(value, x, y);
  }

  function draw() {
    const theta = Number(phase.value) * Math.PI / 180;
    const points = alphabets[modulation.value];
    const samples = received();
    const cx = 210, cy = 205, scale = 105;
    const traceX = 445, traceW = 390, traceH = 125;

    document.getElementById('modclass-phase-value').textContent = `${phase.value}°`;
    document.getElementById('modclass-snr-value').textContent = `${snr.value} dB`;
    ctx.fillStyle = '#10161d';
    ctx.fillRect(0, 0, W, H);

    text('Received constellation', 28, 30, '#ddd', 14);
    text('I', cx + scale + 12, cy + 4);
    text('Q', cx - 4, cy - scale - 12);
    ctx.strokeStyle = 'rgba(210,210,210,0.18)';
    ctx.beginPath();
    ctx.moveTo(cx - scale * 1.45, cy); ctx.lineTo(cx + scale * 1.45, cy);
    ctx.moveTo(cx, cy - scale * 1.45); ctx.lineTo(cx, cy + scale * 1.45);
    ctx.stroke();

    // Faint markers show the noiseless, phase-rotated symbol locations.
    points.forEach(z => {
      const re = z.re * Math.cos(theta) - z.im * Math.sin(theta);
      const im = z.re * Math.sin(theta) + z.im * Math.cos(theta);
      ctx.strokeStyle = colors.ideal;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx + re * scale, cy - im * scale, 6, 0, 2 * Math.PI); ctx.stroke();
    });
    ctx.fillStyle = '#7be0a5';
    samples.forEach(z => {
      ctx.globalAlpha = 0.62;
      ctx.beginPath(); ctx.arc(cx + z.re * scale, cy - z.im * scale, 2.6, 0, 2 * Math.PI); ctx.fill();
    });
    ctx.globalAlpha = 1;
    text('faint rings: ideal rotated symbols', 50, 365, '#aaa', 11);

    text('The same burst over time', traceX, 30, '#ddd', 14);
    const drawTrace = (key, y, color, label) => {
      ctx.strokeStyle = 'rgba(210,210,210,0.14)';
      ctx.beginPath(); ctx.moveTo(traceX, y); ctx.lineTo(traceX + traceW, y); ctx.stroke();
      ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
      samples.forEach((z, index) => {
        const x = traceX + index * traceW / (samples.length - 1);
        const py = y - z[key] * 65;
        index === 0 ? ctx.moveTo(x, py) : ctx.lineTo(x, py);
      });
      ctx.stroke();
      text(label, traceX + 4, y - 73, color, 12);
    };
    drawTrace('re', 130, colors.i, 'I[n]');
    drawTrace('im', 295, colors.q, 'Q[n]');
    text('sample 0', traceX, 375, '#888', 11);
    text('sample 127', traceX + traceW - 57, 375, '#888', 11);
  }

  modulation.addEventListener('change', () => { newBurst(); draw(); });
  phase.addEventListener('input', draw);
  snr.addEventListener('input', draw);
  regenerate.addEventListener('click', () => { newBurst(); draw(); });
  newBurst();
  draw();
})();
</script>

### Training Setup

#### Model Architecture

Our goal is to see if we can only train on a 15 degree rotation band and see if the models generalize after that. There is a real model that is trained on the full rotation degrees just to compare and see what happens.

All four models have similar layers and params. 60,487 for the moment model, 61,655
for the plain complex model, and 60,964 for both real models.

Our models trained on 15 degrees rotation are:

* **Complex:** Processes I and Q together as one complex-valued signal. It learns complex features, then pools their magnitudes so its prediction does not change under one overall signal rotation.
* **Complex with moments:** Uses the same complex feature extractor, then adds phase moments at orders 2, 4, and 8. A phase moment first ignores a point's distance from the origin and keeps only its angle. It then multiplies that angle by 2, 4, or 8 and averages the result across the burst. Points that repeat at a matching spacing reinforce each other: BPSK has a two-way pattern, QPSK has a four-way pattern, and 8PSK has an eight-way pattern. We keep only the size of that average, so rotating the entire constellation does not change the feature.
* **Real Value:** Splits the signal into separate I and Q channels and processes them with a standard real-valued convolutional network. It has to learn rotation invariance from the examples it sees.

Model trained on the full rotation range:

* **Real Value:** The same split-I/Q real model, trained with rotations across the full circle so it can learn that invariance from augmented data.

<div id="modclass-architecture" style="max-width:960px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;background:#10161d;color:#ddd;">
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">
    <button type="button" data-model="complex" style="padding:0.3rem 0.65rem;border:1px solid #4db0ff;border-radius:4px;background:#173248;color:#dceeff;cursor:pointer;">Complex</button>
    <button type="button" data-model="moment" style="padding:0.3rem 0.65rem;border:1px solid #666;border-radius:4px;background:transparent;color:#ddd;cursor:pointer;">Complex + moments</button>
    <button type="button" data-model="real" style="padding:0.3rem 0.65rem;border:1px solid #666;border-radius:4px;background:transparent;color:#ddd;cursor:pointer;">Split-real</button>
  </div>
  <div id="modclass-architecture-note" style="margin:0 0 0.8rem;color:#b9c7d1;font-size:0.88rem;"></div>
  <div id="modclass-architecture-flow" style="display:grid;grid-template-columns:repeat(4,minmax(145px,1fr));gap:0.7rem;"></div>
  <p id="modclass-architecture-detail" style="margin:0.75rem 0 0;color:#b9c7d1;font-size:0.9rem;line-height:1.5;"></p>
</div>

<script>
(() => {
  const root = document.getElementById('modclass-architecture');
  const flow = document.getElementById('modclass-architecture-flow');
  const note = document.getElementById('modclass-architecture-note');
  const detail = document.getElementById('modclass-architecture-detail');
  if (!root || !flow || !note || !detail) return;

  const models = {
    complex: {
      color: '#4db0ff',
      note: 'ComplexModClassifier: rotation-equivariant feature extractor with a magnitude-only invariant head.',
      blocks: [
        ['1. IQ burst', 'One complex channel\n128 samples'],
        ['2. Complex features', '3 x ComplexConv1d + modReLU\nchannels: 1 -> 24 -> 48 -> 48\ntime: 128 -> 64 -> 32 -> 16'],
        ['3. Rotation-safe summary', 'Magnitude mean + standard deviation\n48 channels x 2 = 96 features'],
        ['4. Prediction', '96 -> 128 -> 4\nBPSK, QPSK, 8PSK, 16QAM']
      ],
      detail: 'Complex convolutions have no bias, and modReLU gates magnitude while preserving phase. Pooling only magnitudes makes the final prediction invariant to one global phase rotation.'
    },
    moment: {
      color: '#c78cff',
      note: 'ComplexMomentClassifier: the same equivariant complex feature extractor, with a richer rotation-invariant pooling head.',
      blocks: [
        ['1. IQ burst', 'One complex channel\n128 samples'],
        ['2. Complex features', '3 x ComplexConv1d + modReLU\nchannels: 1 -> 24 -> 48 -> 48\ntime: 128 -> 64 -> 32 -> 16'],
        ['3. Rotation-safe moments', 'Magnitude statistics plus phase moments\norders: 2, 4, and 8\n240 features'],
        ['4. Prediction', '240 -> 48 -> 4\nBPSK, QPSK, 8PSK, 16QAM']
      ],
      detail: 'The extra circular moments retain phase-pattern information that magnitude pooling discards, while their absolute values remain invariant to a global rotation.'
    },
    real: {
      color: '#ff8370',
      note: 'RealModClassifier: a conventional real-valued CNN that receives I and Q as two independent input channels.',
      blocks: [
        ['1. Split IQ burst', 'Two real channels\nI[128] and Q[128]'],
        ['2. Real features', '3 x Conv1d + BatchNorm + ReLU\nchannels: 2 -> 32 -> 64 -> 64\ntime: 128 -> 64 -> 32 -> 16'],
        ['3. Statistics summary', 'Mean + standard deviation\n64 channels x 2 = 128 features'],
        ['4. Prediction', '128 -> 128 -> 4\nBPSK, QPSK, 8PSK, 16QAM']
      ],
      detail: 'The real network has no built-in global-phase symmetry. We train it either on rotations within +/-15 degrees (real_narrow) or across the full circle (real_full) to measure the cost of learning that invariance from data.'
    }
  };

  function show(name) {
    const model = models[name];
    note.textContent = model.note;
    detail.textContent = model.detail;
    flow.replaceChildren();
    model.blocks.forEach((block, index) => {
      const card = document.createElement('div');
      card.style.cssText = `min-width:0;padding:0.8rem;border:1px solid ${model.color};border-radius:6px;background:#17212a;`;
      const title = document.createElement('strong');
      title.textContent = block[0];
      title.style.cssText = `display:block;margin-bottom:0.45rem;color:${model.color};font-size:0.95rem;`;
      const body = document.createElement('span');
      body.textContent = block[1];
      body.style.cssText = 'white-space:pre-line;color:#d4dce2;font:0.88rem/1.5 sans-serif;';
      card.append(title, body);
      flow.appendChild(card);
    });
    root.querySelectorAll('button[data-model]').forEach(button => {
      const active = button.dataset.model === name;
      button.style.borderColor = active ? model.color : '#666';
      button.style.background = active ? '#24333e' : 'transparent';
      button.style.color = active ? '#fff' : '#ddd';
    });
  }

  root.querySelectorAll('button[data-model]').forEach(button => button.addEventListener('click', () => show(button.dataset.model)));
  show('complex');
})();
</script>

#### Dataset

- **Signal:** Complex baseband IQ (`complex64`), 128 symbols per burst, and unit-average-power constellations.
- **Classes:** BPSK, QPSK, 8PSK, and 16QAM; balanced and shuffled with 3,000 training bursts per class.
- **Splits:** 12,000 training bursts, 4,000 in-distribution validation bursts, and 4,000 full-circle validation bursts.
- **Channel effects:** One uniformly sampled carrier phase per burst plus AWGN at 10 dB SNR, scaled to the burst power.
- **Rotation ranges:** Narrow runs use +/-15 degrees; `real_full` uses +/-180 degrees.
- **Reproducibility:** Random seed 7, with separate seed offsets for each split.

#### Training Params

- **Runs:** `complex_narrow`, `complex_moment`, `real_narrow`, and `real_full`.
- **Schedule:** 25 epochs, batch size 256, and shuffled training batches.
- **Optimization:** Adam with learning rate 0.001, weight decay 0.0001, and gradient-norm clipping at 5.0.
- **Objective:** Four-class cross entropy; the checkpoint with the best full-circle validation score is saved as `best_model.pt`.

### Results

Model checkpoints are evaluated with a full-circle rotation sweep
in 5 degree steps, a per-modulation rotation sweep, a full-circle SNR sweep,
and confusion matrices at 0 and 90 degrees. Rotations use 10 dB SNR and
2,000 generated test bursts at each angle. SNR use 4,000 bursts at each
noise level.

| Model | Training rotations | Accuracy at 0 degrees | Accuracy at 90 degrees | Lowest accuracy across 360 degrees | Mean accuracy across 360 degrees |
| --- | --- | ---: | ---: | ---: | ---: |
| Complex + moments | +/-15 degrees | 98.6% | 98.3% | 97.9% | **98.4%** |
| Complex | +/-15 degrees | 81.4% | 81.2% | 80.2% | 81.0% |
| Real narrow | +/-15 degrees | 97.1% | 72.2% | 45.9% | 63.5% |
| Real full | +/-180 degrees | 72.0% | 72.2% | 70.6% | 72.0% |

The moment model stays nearly flat over every unseen rotation. The narrow real
model is most accurate at 0 degrees where it trained,
and then crumbles as the burst rotates away from that band: 97.1% at 0
degrees, 72.2% at 90 degrees, and 45.9% at its worst angle. It memorized the
band it saw instead of learning the symmetry. Training the real model across
the whole circle removes that collapse but it still does not match the moment model.

<p><strong>Overall rotation accuracy.</strong> The notebook evaluates every 5 degrees from -180 to +180 degrees.</p>
<img src="{{ '/assets/images/modclass_rotation_generalization.png' | relative_url }}" alt="Classification accuracy versus carrier phase rotation for all four models" style="width:100%;height:auto;border-radius:6px;">

<p><strong>Accuracy by modulation type.</strong> These panels show where each modulation contributes to the narrow real model's rotation sensitivity.</p>
<img src="{{ '/assets/images/modclass_rotation_per_modulation.png' | relative_url }}" alt="Per-modulation classification accuracy versus carrier phase rotation" style="width:100%;height:auto;border-radius:6px;">

| Model | 20 dB SNR | 10 dB SNR | 5 dB SNR | 0 dB SNR |
| --- | ---: | ---: | ---: | ---: |
| Complex + moments | 97.1% | **98.3%** | 39.3% | 25.0% |
| Complex | 83.8% | 81.9% | 29.9% | 25.0% |
| Real narrow | 57.2% | 62.7% | 29.8% | 25.0% |
| Real full | 68.5% | 71.5% | **44.6%** | 25.0% |

At 0 dB and below, all models are at the four-class chance level of 25%.

A few notes that are worth pointing out is how some models do better at 10db than 20db. This is due to having the models train at 10db so it has learned that noise pattern better than the others. All the models take a hit at 5db for multiple reasons. The models are trained at 10db for one and 5db is difficult since the signal blends in with the noise.

The sweep also shows that rotation invariance and noise robustness are separate
problems. The moment model is strongest at the 10 dB condition it trained on,
while the full-rotation real model holds up better at 5 dB.

<p><strong>Noise sweep.</strong> Each point uses rotations uniformly distributed across the full circle.</p>
<img src="{{ '/assets/images/modclass_snr_sweep.png' | relative_url }}" alt="Full-circle modulation classification accuracy at different SNR values" style="width:100%;height:auto;border-radius:6px;">

<p><strong>Confusion matrices.</strong> Rows are true modulation labels and columns are predicted labels. The 90 degree column is outside the narrow training band.</p>
<img src="{{ '/assets/images/modclass_confusion_in_vs_ood.png' | relative_url }}" alt="Confusion matrices comparing in-distribution and out-of-distribution carrier-phase rotations" style="width:100%;height:auto;border-radius:6px;">

### Explaining the Results

Our hypothesis was that complex models generalize better from less data. The
results support this but that isn't the full

While the complex model performed better at generalization when compared to the real narrow model this does not mean that the complex math was the end all be all. The real gains came from the moment model that built a structure to capture the symmetry. Calculating this phase angles and averaging them helped to provide the best model for generalizing beyond the trained 15 degrees of rotation.

This does not mean a real model trained on the full range of degrees could not also learn this. The full real model does the best at 5db potentially due to be trained on the full degrees. The complex versions are able to generalize with less training and on a smaller degree of rotation.

The SNR sweep is measuring a different axis entirely. Lower SNR makes the
signal harder to separate from the noise since it blends into it, which is a
robustness problem rather than a symmetry problem.

### Limitations

Keep in mind that this problem in the real world is harder. Signals come faded, contain frequency offsets, timing offsets, imbalances and more. Our example proves out the idea for how we can better utilize our data and training efficiency by building known symmetries into the model.

This is also a single seed at a single training SNR, and the data is generated
with TorchSig rather than captured off the air.

## RF Fingerprinting

The next task we will tackle is a bit more challenging. We will try to do RF fingerprinting, which is where you try to tell which emitter sent the signal. Even if two emitters send the same signal, no manufacturer produces an antenna and apparatus to be exactly the same. They are allowed a margin of error. Our task is to detect the exact emitter that sent a signal even when two emitters send the same message.

In this experiment, every emitter sends QPSK. The information in the message is
not the label. Instead, the label is the small and repeatable distortion added
by the transmitter hardware: I/Q imbalance, oscillator phase noise, and power
amplifier nonlinearity. Gain, carrier phase, and channel noise change for every
received capture, so the classifier must learn the transmitter rather than the
channel.

<div id="rf-fingerprint-demo" style="max-width:960px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;background:#10161d;color:#ddd;">
  <canvas id="rf-fingerprint-canvas" width="920" height="390" style="width:100%;height:auto;display:block;border-radius:6px;background:#10161d;"></canvas>
  <div style="margin-top:0.7rem;display:flex;gap:0.7rem;flex-wrap:wrap;align-items:center;font-size:0.85rem;">
    <label style="display:flex;gap:0.35rem;align-items:center;">channel phase
      <input id="rf-fingerprint-phase" type="range" min="-180" max="180" step="5" value="25" style="width:105px;">
      <span id="rf-fingerprint-phase-value" style="min-width:3.5em;">25°</span>
    </label>
    <label style="display:flex;gap:0.35rem;align-items:center;">SNR
      <input id="rf-fingerprint-snr" type="range" min="0" max="30" step="1" value="18" style="width:90px;">
      <span id="rf-fingerprint-snr-value" style="min-width:3.5em;">18 dB</span>
    </label>
    <button id="rf-fingerprint-regenerate" type="button" style="padding:0.25rem 0.6rem;border:1px solid #4db0ff;border-radius:4px;background:transparent;color:#4db0ff;cursor:pointer;">new shared message</button>
  </div>
  <p style="margin:0.65rem 0 0;color:#aaa;font-size:0.82rem;">Both panels begin with the same QPSK symbols. The small, persistent differences are the synthetic transmitter fingerprints; the global rotation and noise are channel effects that change each capture.</p>
</div>

<script>
(() => {
  const canvas = document.getElementById('rf-fingerprint-canvas');
  const phase = document.getElementById('rf-fingerprint-phase');
  const snr = document.getElementById('rf-fingerprint-snr');
  const regenerate = document.getElementById('rf-fingerprint-regenerate');
  if (!canvas || !phase || !snr || !regenerate) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const emitters = [
    { name: 'Emitter A', color: '#4db0ff', gain: -3.2, skew: -0.18, phaseNoise: 0.010, backoff: 2.8, ampm: -0.22 },
    { name: 'Emitter B', color: '#ff9f5b', gain: 3.2, skew: 0.18, phaseNoise: 0.055, backoff: 8.8, ampm: 0.22 }
  ];
  let symbols = [];

  function gaussian() {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function newMessage() {
    symbols = Array.from({ length: 32 }, () => {
      const angle = Math.PI / 4 + Math.floor(Math.random() * 4) * Math.PI / 2;
      return { re: Math.cos(angle), im: Math.sin(angle) };
    });
  }

  function capture(emitter) {
    const samples = [];
    const gain = Math.pow(10, emitter.gain / 40);
    const qGain = 1 / gain;
    const channelPhase = Number(phase.value) * Math.PI / 180;
    const noiseSigma = Math.sqrt(1 / (2 * Math.pow(10, Number(snr.value) / 10)));
    let wander = 0;
    for (let k = 0; k < symbols.length; k++) {
      const a = symbols[k], b = symbols[(k + 1) % symbols.length];
      for (let n = 0; n < 4; n++) {
        const t = n / 4;
        // Inter-symbol samples expose the amplifier's amplitude-dependent error.
        let re = a.re * (1 - t) + b.re * t;
        let im = a.im * (1 - t) + b.im * t;
        re *= gain;
        im *= qGain;
        const rotatedI = re * Math.cos(-emitter.skew / 2) - im * Math.sin(-emitter.skew / 2);
        const rotatedQ = re * Math.sin(emitter.skew / 2) + im * Math.cos(emitter.skew / 2);
        const magnitude = Math.hypot(rotatedI, rotatedQ);
        const compression = 1 / (1 + magnitude * magnitude / emitter.backoff);
        wander += gaussian() * emitter.phaseNoise;
        const paPhase = emitter.ampm * magnitude * magnitude / (1 + magnitude * magnitude);
        const angle = channelPhase + wander + paPhase;
        samples.push({
          re: compression * (rotatedI * Math.cos(angle) - rotatedQ * Math.sin(angle)) + noiseSigma * gaussian(),
          im: compression * (rotatedI * Math.sin(angle) + rotatedQ * Math.cos(angle)) + noiseSigma * gaussian()
        });
      }
    }
    return samples;
  }

  function label(value, x, y, color = '#aaa', size = 12) {
    ctx.fillStyle = color;
    ctx.font = `${size}px sans-serif`;
    ctx.fillText(value, x, y);
  }

  function drawPanel(samples, emitter, cx) {
    const cy = 202, scale = 92;
    label(emitter.name, cx - 85, 30, emitter.color, 15);
    label('same message + fixed hardware fingerprint', cx - 85, 49, '#aaa', 11);
    ctx.strokeStyle = 'rgba(210,210,210,0.18)';
    ctx.beginPath();
    ctx.moveTo(cx - 125, cy); ctx.lineTo(cx + 125, cy);
    ctx.moveTo(cx, cy - 125); ctx.lineTo(cx, cy + 125);
    ctx.stroke();
    ctx.fillStyle = emitter.color;
    samples.forEach(z => {
      ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.arc(cx + z.re * scale, cy - z.im * scale, 2.2, 0, 2 * Math.PI); ctx.fill();
    });
    ctx.globalAlpha = 1;
    label('I', cx + 132, cy + 4, '#aaa', 11);
    label('Q', cx - 4, cy - 135, '#aaa', 11);
    label(`I/Q gain: ${emitter.gain > 0 ? '+' : ''}${emitter.gain.toFixed(1)} dB`, cx - 105, 350, '#bbb', 11);
    label(`phase noise: ${(emitter.phaseNoise * 100).toFixed(1)}%`, cx - 105, 367, '#bbb', 11);
  }

  function draw() {
    document.getElementById('rf-fingerprint-phase-value').textContent = `${phase.value}°`;
    document.getElementById('rf-fingerprint-snr-value').textContent = `${snr.value} dB`;
    ctx.fillStyle = '#10161d'; ctx.fillRect(0, 0, W, H);
    label('Shared QPSK payload', 34, 30, '#ddd', 14);
    label('The bits are intentionally not a device clue', 34, 49, '#aaa', 11);
    ctx.strokeStyle = 'rgba(210,210,210,0.18)';
    ctx.beginPath(); ctx.moveTo(115, 86); ctx.lineTo(115, 316); ctx.moveTo(0, 201); ctx.lineTo(230, 201); ctx.stroke();
    ctx.fillStyle = '#7be0a5';
    symbols.forEach(z => { ctx.beginPath(); ctx.arc(115 + z.re * 75, 201 - z.im * 75, 4, 0, 2 * Math.PI); ctx.fill(); });
    label('ideal symbols', 72, 345, '#aaa', 11);
    ctx.strokeStyle = 'rgba(210,210,210,0.22)'; ctx.beginPath(); ctx.moveTo(240, 195); ctx.lineTo(275, 195); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.beginPath(); ctx.moveTo(275, 195); ctx.lineTo(266, 190); ctx.lineTo(266, 200); ctx.fill();
    drawPanel(capture(emitters[0]), emitters[0], 410);
    drawPanel(capture(emitters[1]), emitters[1], 700);
  }

  phase.addEventListener('input', draw);
  snr.addEventListener('input', draw);
  regenerate.addEventListener('click', () => { newMessage(); draw(); });
  newMessage(); draw();
})();
</script>

### Training Setup

#### Model Architecture

<div id="rf-fingerprint-architecture" style="max-width:960px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;background:#10161d;color:#ddd;">
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">
    <button type="button" data-model="real" style="padding:0.3rem 0.65rem;border:1px solid #ff8370;border-radius:4px;background:#3a2523;color:#fff;cursor:pointer;">Real CNN</button>
    <button type="button" data-model="complex" style="padding:0.3rem 0.65rem;border:1px solid #666;border-radius:4px;background:transparent;color:#ddd;cursor:pointer;">Complex CNN</button>
  </div>
  <div id="rf-fingerprint-architecture-flow" style="display:grid;grid-template-columns:repeat(4,minmax(145px,1fr));gap:0.7rem;"></div>
</div>

<script>
(() => {
  const root = document.getElementById('rf-fingerprint-architecture');
  const flow = document.getElementById('rf-fingerprint-architecture-flow');
  if (!root || !flow) return;
  const models = {
    real: { color: '#ff8370', blocks: [['1. Split IQ burst', 'I[n] and Q[n]\n2 real channels'], ['2. Real features', '3 x Conv1d + BatchNorm + ReLU\n2 -> 24 -> 48 -> 64 channels'], ['3. Statistics', 'Mean + standard deviation + maximum\n192 features'], ['4. Emitter prediction', '192 -> 64 -> N emitters']] },
    complex: { color: '#4db0ff', blocks: [['1. Complex IQ burst', 'One complex channel'], ['2. Complex features', '3 x ComplexConv1d + ComplexBatchNorm + modReLU\n1 -> 20 -> 30 -> 40 channels'], ['3. Complex statistics', 'Real, imaginary, and magnitude statistics\n240 features'], ['4. Emitter prediction', '240 -> 64 -> N emitters']] }
  };
  function show(name) {
    const model = models[name]; flow.replaceChildren();
    model.blocks.forEach(block => {
      const card = document.createElement('div'); card.style.cssText = `min-width:0;padding:0.8rem;border:1px solid ${model.color};border-radius:6px;background:#17212a;`;
      const title = document.createElement('strong'); title.textContent = block[0]; title.style.cssText = `display:block;margin-bottom:0.45rem;color:${model.color};font-size:0.95rem;`;
      const body = document.createElement('span'); body.textContent = block[1]; body.style.cssText = 'white-space:pre-line;color:#d4dce2;font:0.88rem/1.5 sans-serif;';
      card.append(title, body); flow.appendChild(card);
    });
    root.querySelectorAll('button[data-model]').forEach(button => { const active = button.dataset.model === name; button.style.borderColor = active ? model.color : '#666'; button.style.background = active ? '#24333e' : 'transparent'; button.style.color = active ? '#fff' : '#ddd'; });
  }
  root.querySelectorAll('button[data-model]').forEach(button => button.addEventListener('click', () => show(button.dataset.model)));
  show('real');
})();
</script>

#### Dataset

- **Signal:** QPSK complex baseband IQ, 256 samples per capture.
- **Classes:** 4 synthetic emitters.
- **Hardware fingerprint:** Fixed differential I/Q gain and phase imbalance, oscillator phase noise, and power-amplifier nonlinearity per emitter.
- **Per-capture channel effects:** Independent payload, gain (0.8 to 1.2), phase (-0.05 to 0.05 rad), and AWGN; bursts normalized to unit power.
- **Training split:** 400 captures per emitter, 1,600 total, at 20 dB SNR.
- **Test split:** 120 held-out captures per emitter, 480 total, at 20 dB SNR.

<p><strong>Hardware-impaired constellations.</strong> Two of the four emitters. The same QPSK waveform develops different point-cloud shapes after each emitter's fixed hardware effects and capture variation.</p>
<img src="{{ '/assets/images/rf_fingerprint_constellations.png' | relative_url }}" alt="Constellations from two synthetic QPSK emitters with different hardware fingerprints" style="width:100%;height:auto;border-radius:6px;">

#### Training Params

- **Models:** Real CNN and complex CNN; both use 3 stride-2 convolutional layers and a 64-value embedding.
- **Epochs:** 17.
- **Batch size:** 128.
- **Optimizer:** Adam.
- **Learning rate:** 0.001.
- **Objective:** Four-class cross entropy.

### Results

Both the Real and Complex models contain almost the same amount of params.

| Model | Real scalar parameters | Test accuracy | Worst-emitter accuracy |
| --- | ---: | ---: | ---: |
| Real CNN | 34,476 | 100.0% | 100.0% |
| Complex CNN | 34,964 | 99.8% | 99.2% |

<p><strong>Closed-set confusion matrices.</strong> Rows are the true emitter identities and columns are predicted identities for the held-out captures.</p>
<img src="{{ '/assets/images/rf_fingerprint_confusion_matrices.png' | relative_url }}" alt="Real and complex CNN confusion matrices for four synthetic RF emitters" style="width:100%;height:auto;border-radius:6px;">

The models also perform well at different noise levels as seen below.

| Model | 6 dB SNR | 10 dB SNR | 14 dB SNR | 18 dB SNR | 22 dB SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real CNN | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Complex CNN | 96.5% | 99.5% | 99.5% | 100.0% | 100.0% |

<p><strong>Noise sweep.</strong> The trained models are evaluated on newly generated captures at each SNR without retraining.</p>
<img src="{{ '/assets/images/rf_fingerprint_snr_sweep.png' | relative_url }}" alt="RF fingerprinting accuracy at different signal-to-noise ratios for real and complex CNNs" style="width:100%;height:auto;border-radius:6px;">

The open-set test used two emitters the models had never seen before. Both models scored similar and the complex model just edged out a higher score, but not enough to draw any major conclusions.

| Model | Unknown-emitter AUROC |
| --- | ---: |
| Real CNN | 0.656 |
| Complex CNN | 0.699 |

<p><strong>Learned embeddings.</strong> The 64-value encoder outputs are projected to two dimensions with PCA; color identifies the true emitter.</p>
<img src="{{ '/assets/images/rf_fingerprint_embeddings.png' | relative_url }}" alt="PCA projections of real and complex RF fingerprint encoder embeddings" style="width:100%;height:auto;border-radius:6px;">


### Explaining the Results

For modulation classification the rotation-invariant complex model clearly won. Here the real and complex models are effectively tied, and that difference lies in the geometry and task.

The modulation task had a symmetry worth exploiting: a global carrier phase rotation changes the input but not the label. RF fingerprinting is not the same. The label is a set of arbitrary device-specific distortions, and there is no clean geometric transformation the architecture can be made invariant to. Without a symmetry to build in, complex layers don't offer the help that they did for modulation classification.

- Both models receive the same full IQ waveform. A real CNN can learn relationships between I and Q when they are supplied as two channels.
- The hardware fingerprints are arbitrary device-specific distortions, not a simple global-rotation symmetry where complex-valued processing has a built-in advantage.
- Differential I/Q imbalance has a conjugate structure that the complex model can represent directly, but a sufficiently capable real CNN can represent the same transformation.
- Both models are capacity-matched at roughly 34.5k real scalar parameters, and this controlled 4-emitter task is close to an accuracy ceiling, so there is little headroom for either to separate.

### Limitations

This RF fingerprint example is heavily simplified compared to the real world. This problem may show a different story if we took the time to make it more intricate and more realistic. RF fingerprinting is challenging and there are many ways to scale this experiment to include more distortions and expand the open and closed set of emitters.

This experiment only trains on 4 emitters, which is very small, and all of the data is synthetic, generated with `torchsig`. Real emitters will not play as neat and tidy as our examples. There are real RF fingerprinting datasets out there, but for this blog post we emulated the problem instead. There are many distortions we have not applied, and no doppler shifting or frequency offsets. These results are here to show that complex models are not automatically better than real models. You have to understand the problem and know when to reach for them.

## Conclusion

We have taken our knowledge of Wirtinger derivatives and applied them with complex models to test how they compare with real models. The gist of it is that when a problem has a global geometric symmetry, complex layers give you a natural way to build that symmetry into the architecture, and you get generalization the real model has to buy with data. They are not always the best. For fingerprinting there are so many device-specific modifications that you lose that advantage entirely, because there is no global geometric structure left to capture.

The modulation experiment shows that complex arithmetic on its own was not enough. The plain complex model lost to the real model inside its training band. The gain came from the invariant pooling head built on top of those complex features. Complex layers make that kind of head easy to express.

The important part is understanding when a problem actually has a symmetry worth encoding, and recognizing that when it doesn't, a well-built real model will do fine.

[Modulation Classification Notebook](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/complex_vs_real_nn/modulation_classification/complex_vs_real_nn.ipynb)

[RF Fingerprinting Notebook](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/complex_vs_real_nn/rf_fingerprinting/rf_fingerprint.ipynb)
