---
layout: post
title: "Complex Numbers and Deep Learning Part 1"
date: 2026-07-12
categories: ML
---

No one knows what an imaginary number is. It's like asking ML people how this algorithm is meant to work outside of a notebook. That reality just does not exist. Imaginary numbers are very powerful for certain areas of engineering. Deep learning algorithms can also use imaginary numbers but often do not due to limitations of back propagation. There are ways to get around this and you can keep the natural relationship between real numbers and imaginary. Keeping this relationship together can have the model learn more efficiently and generalize better. Before we get into all that we need to understand what an imaginary number is and what a complex number is.

## Complex Numbers

Imaginary numbers are a bad name. Imaginary makes it seem like it's not real but it is. This is in some ways hard to grasp like negative numbers were hard to grasp for many in mathematics. Which makes sense because if you have 3 apples and you take 4 apples away you now have -1 apples. How in the world can you have -1 apples? It sounds absurd but that doesn't mean negative numbers are not real. In banking it makes a lot of sense. You are in the hole and have negative dollars. You gotta get out of that hole.

World most trusted source Wikipedia defines an imaginary number as: "the product of a real number and the imaginary unit $i$, which is defined by its property $i^2 = -1$."

An imaginary number is then any real multiple of $i$, written as:

$$
bi \quad \text{where} \quad b \in \mathbb{R}
$$

A complex number combines a real part and an imaginary part:

$$
z = a + bi \quad \text{where} \quad a,b \in \mathbb{R}
$$

These definitions don't help at all. People can read that definition all day long but it won't click.

I am approaching this concept from an engineering and ML perspective. I am not a pure mathematician who would probably disagree with this statement.

I view imaginary and complex numbers as **rotations**. In the practical sense the problems that deal with complex numbers have some relationship with rotations or cycles.

What I mean by that is in the practical world when you multiply a real number by an imaginary number you are rotating 90 degrees.

Multiplying a complex number by $i$ rotates it by $90^\circ$ counterclockwise:

$$
i(a + bi) = -b + ai
$$

In matrix form, this is the same as applying a 2D rotation by $90^\circ$:

$$
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
=
\begin{bmatrix}
-b \\
a
\end{bmatrix}
$$

### Multiply by $i$ Visualization

This animation keeps the rotation fixed at $90^\circ$ so you can see how
$(a,b)$ maps to $(-b,a)$.

<div id="complex-i-rotation-demo" style="max-width:420px;margin:1rem auto;padding:0.75rem;border:1px solid #444;border-radius:8px;">
  <canvas id="complex-i-rotation-canvas" width="380" height="300" style="width:100%;height:auto;display:block;"></canvas>
  <div style="margin-top:0.6rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="complex-i-a">a</label>
    <input id="complex-i-a" type="range" min="-2" max="2" step="0.1" value="1.0">
    <span id="complex-i-a-value">1.0</span>

    <label for="complex-i-b">b</label>
    <input id="complex-i-b" type="range" min="-2" max="2" step="0.1" value="1.0">
    <span id="complex-i-b-value">1.0</span>
  </div>
  <div style="margin-top:0.5rem;font-size:0.95rem;">
    <strong>Mapping:</strong> $(a,b) \rightarrow (-b,a)$
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('complex-i-rotation-canvas');
  const aInput = document.getElementById('complex-i-a');
  const bInput = document.getElementById('complex-i-b');
  const aValue = document.getElementById('complex-i-a-value');
  const bValue = document.getElementById('complex-i-b-value');
  if (!canvas || !aInput || !bInput || !aValue || !bValue) return;

  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const scale = 65;
  let t = 0;

  function drawAxes() {
    ctx.strokeStyle = '#7a7a7a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, cy);
    ctx.lineTo(canvas.width - 20, cy);
    ctx.moveTo(cx, 20);
    ctx.lineTo(cx, canvas.height - 20);
    ctx.stroke();
  }

  function drawArrow(x, y, color, width) {
    const px = cx + x * scale;
    const py = cy - y * scale;
    const ang = Math.atan2(py - cy, px - cx);
    const head = 9;

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - head * Math.cos(ang - Math.PI / 6), py - head * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(px - head * Math.cos(ang + Math.PI / 6), py - head * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function frame() {
    const a = Number(aInput.value);
    const b = Number(bInput.value);
    const rx = -b;
    const ry = a;

    aValue.textContent = a.toFixed(1);
    bValue.textContent = b.toFixed(1);

    const phase = 0.5 + 0.5 * Math.sin(t);
    const ix = a + (rx - a) * phase;
    const iy = b + (ry - b) * phase;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();

    drawArrow(a, b, '#9aa4b2', 2);
    drawArrow(rx, ry, '#4aa3ff', 2);
    drawArrow(ix, iy, '#f4b942', 2.5);

    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#9aa4b2';
    ctx.fillText('Original (a,b)', 16, 20);
    ctx.fillStyle = '#4aa3ff';
    ctx.fillText('i(a+bi) = (-b,a)', 16, 38);
    ctx.fillStyle = '#f4b942';
    ctx.fillText('Animated transition', 16, 56);

    t += 0.04;
    requestAnimationFrame(frame);
  }

  frame();
})();
</script>

### Polar Form

A complex number can be written in two equivalent ways:

- **Cartesian form:** $z = a + bi$ (horizontal plus vertical parts).
- **Polar form:** $z = re^{i\theta}$ (length plus direction).

They represent the same point in the complex plane:

$$
z = a + bi = r(\cos\theta + i\sin\theta) = re^{i\theta}
$$

where

$$
r = |z| = \sqrt{a^2 + b^2}, \qquad \theta = \arg(z) = \operatorname{atan2}(b,a)
$$

So $r$ tells you how far from the origin, and $\theta$ tells you the direction from the positive real axis.

Multiplying by $re^{i\theta}$ does two things at once: scale by $r$ and rotate by $\theta$.

Angles are periodic, so these all point the same direction:

$$
\theta,\ \theta + 2\pi,\ \theta + 4\pi,\ \ldots,\ \theta + 2\pi k \quad (k \in \mathbb{Z})
$$

### Polar Form Visualization

This animation shows the same complex number in both forms: $z = a + bi$ and $z = re^{i\theta}$.

<div id="polar-form-demo" style="max-width:420px;margin:1rem auto;padding:0.75rem;border:1px solid #444;border-radius:8px;">
  <canvas id="polar-form-canvas" width="380" height="300" style="width:100%;height:auto;display:block;"></canvas>
  <div style="margin-top:0.6rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="polar-r">r</label>
    <input id="polar-r" type="range" min="0.2" max="2.5" step="0.1" value="1.4">
    <span id="polar-r-value">1.4</span>

    <label for="polar-theta">theta</label>
    <input id="polar-theta" type="range" min="-180" max="180" step="1" value="45">
    <span id="polar-theta-value">45deg</span>
  </div>
  <div style="margin-top:0.5rem;font-size:0.95rem;">
    <strong>Live values:</strong> <span id="polar-live-values"></span>
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('polar-form-canvas');
  const rInput = document.getElementById('polar-r');
  const thetaInput = document.getElementById('polar-theta');
  const rValue = document.getElementById('polar-r-value');
  const thetaValue = document.getElementById('polar-theta-value');
  const liveValues = document.getElementById('polar-live-values');
  if (!canvas || !rInput || !thetaInput || !rValue || !thetaValue || !liveValues) return;

  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const scale = 65;

  function drawAxes() {
    ctx.strokeStyle = '#7a7a7a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, cy);
    ctx.lineTo(canvas.width - 20, cy);
    ctx.moveTo(cx, 20);
    ctx.lineTo(cx, canvas.height - 20);
    ctx.stroke();
  }

  function drawArrow(x, y, color, width) {
    const px = cx + x * scale;
    const py = cy - y * scale;
    const ang = Math.atan2(py - cy, px - cx);
    const head = 9;

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - head * Math.cos(ang - Math.PI / 6), py - head * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(px - head * Math.cos(ang + Math.PI / 6), py - head * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function drawProjection(x, y) {
    const px = cx + x * scale;
    const py = cy - y * scale;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#5b6470';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, cy);
    ctx.moveTo(px, py);
    ctx.lineTo(cx, py);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawAngle(thetaRad) {
    ctx.strokeStyle = '#f4b942';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 26, 0, -thetaRad, thetaRad > 0);
    ctx.stroke();
  }

  function render() {
    const r = Number(rInput.value);
    const thetaDeg = Number(thetaInput.value);
    const thetaRad = thetaDeg * Math.PI / 180;
    const a = r * Math.cos(thetaRad);
    const b = r * Math.sin(thetaRad);

    rValue.textContent = r.toFixed(1);
    thetaValue.textContent = thetaDeg.toFixed(0) + 'deg';
    liveValues.textContent =
      'a=' + a.toFixed(2) + ', b=' + b.toFixed(2) + ', z=' + a.toFixed(2) + '+' + b.toFixed(2) + 'i';

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();
    drawProjection(a, b);
    drawAngle(thetaRad);
    drawArrow(a, b, '#4aa3ff', 2.5);

    ctx.fillStyle = '#9aa4b2';
    ctx.font = '12px sans-serif';
    ctx.fillText('z = a + bi', 16, 20);
    ctx.fillText('z = re^{i theta}', 16, 38);
    ctx.fillStyle = '#f4b942';
    ctx.fillText('theta arc', 16, 56);
  }

  rInput.addEventListener('input', render);
  thetaInput.addEventListener('input', render);
  render();
})();
</script>

### Rotation Visualization

Here is an animation that shows the original vector and the rotated version.

<div id="complex-rotation-demo" style="max-width:420px;margin:1rem auto;padding:0.75rem;border:1px solid #444;border-radius:8px;">
  <canvas id="complex-rotation-canvas" width="380" height="300" style="width:100%;height:auto;display:block;"></canvas>
  <div style="margin-top:0.6rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="complex-x">x</label>
    <input id="complex-x" type="range" min="-2" max="2" step="0.1" value="1.0">
    <span id="complex-x-value">1.0</span>

    <label for="complex-y">y</label>
    <input id="complex-y" type="range" min="-2" max="2" step="0.1" value="1.0">
    <span id="complex-y-value">1.0</span>

    <label for="complex-angle">angle</label>
    <input id="complex-angle" type="range" min="-180" max="180" step="1" value="45">
    <span id="complex-angle-value">45deg</span>
  </div>
  <div style="margin-top:0.5rem;font-size:0.95rem;">
    <strong>Current rotation:</strong> <span id="complex-rotation-angle">0</span> deg
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('complex-rotation-canvas');
  const angleLabel = document.getElementById('complex-rotation-angle');
  const xInput = document.getElementById('complex-x');
  const yInput = document.getElementById('complex-y');
  const angleInput = document.getElementById('complex-angle');
  const xValue = document.getElementById('complex-x-value');
  const yValue = document.getElementById('complex-y-value');
  const angleValue = document.getElementById('complex-angle-value');

  if (!canvas || !angleLabel || !xInput || !yInput || !angleInput || !xValue || !yValue || !angleValue) return;

  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const scale = 65;
  let time = 0;

  function toNumber(v) {
    return Number(v);
  }

  function drawArrow(x, y, color, width) {
    const px = cx + x * scale;
    const py = cy - y * scale;
    const angle = Math.atan2(py - cy, px - cx);
    const head = 9;

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - head * Math.cos(angle - Math.PI / 6), py - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(px - head * Math.cos(angle + Math.PI / 6), py - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function drawAxes() {
    ctx.strokeStyle = '#7a7a7a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, cy);
    ctx.lineTo(canvas.width - 20, cy);
    ctx.moveTo(cx, 20);
    ctx.lineTo(cx, canvas.height - 20);
    ctx.stroke();

    ctx.fillStyle = '#7a7a7a';
    ctx.font = '12px sans-serif';
    ctx.fillText('Re', canvas.width - 35, cy - 6);
    ctx.fillText('Im', cx + 8, 30);
  }

  function drawGridCircle(rUnits) {
    ctx.strokeStyle = '#3b3b3b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, rUnits * scale, 0, Math.PI * 2);
    ctx.stroke();
  }

  function drawArc(startAngle, endAngle) {
    ctx.strokeStyle = '#f4b942';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 30, -startAngle, -endAngle, endAngle > startAngle);
    ctx.stroke();
  }

  function rotate(x, y, angleRad) {
    const c = Math.cos(angleRad);
    const s = Math.sin(angleRad);
    return {
      x: x * c - y * s,
      y: x * s + y * c
    };
  }

  function updateLabels(x, y, angleDeg) {
    xValue.textContent = x.toFixed(1);
    yValue.textContent = y.toFixed(1);
    angleValue.textContent = angleDeg.toFixed(0) + 'deg';
  }

  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();
    drawGridCircle(1);
    drawGridCircle(2);

    const x = toNumber(xInput.value);
    const y = toNumber(yInput.value);
    const targetDeg = toNumber(angleInput.value);
    const targetRad = targetDeg * Math.PI / 180;
    updateLabels(x, y, targetDeg);

    const phase = 0.5 + 0.5 * Math.sin(time);
    const currentRad = targetRad * phase;
    const currentDeg = currentRad * 180 / Math.PI;

    const baseAngle = Math.atan2(y, x);
    const rotated = rotate(x, y, currentRad);

    drawArrow(x, y, '#9aa4b2', 2);
    drawArrow(rotated.x, rotated.y, '#4aa3ff', 2.5);
    drawArc(baseAngle, baseAngle + currentRad);

    ctx.fillStyle = '#9aa4b2';
    ctx.font = '12px sans-serif';
    ctx.fillText('Original', 18, 20);
    ctx.fillStyle = '#4aa3ff';
    ctx.fillText('Rotated', 18, 38);
    ctx.fillStyle = '#f4b942';
    ctx.fillText('Angle', 18, 56);

    angleLabel.textContent = currentDeg.toFixed(1);
    time += 0.03;
    requestAnimationFrame(frame);
  }

  frame();
})();
</script>

I hope defining these terms and showing these animations help get the point across that complex numbers deal with rotating and scaling vectors.

## Complex Numbers and ML

So if complex numbers contain rotation information then why are complex neural networks not that popular? It has to do with how back propagation works. Complex numbers do not propagate gradients well and the hardware is not optimized for this process

### Holomorphic vs Non-Holomorphic

For people new to this or those who forgot what they learned in calculus class like me, here is a simple way to define the terms:

- **Holomorphic:** one consistent complex derivative in every direction.
- **Non-holomorphic:** derivative estimate changes with direction.

The derivative test is:

$$
f'(z) = \lim_{h\to 0} \frac{f(z+h)-f(z)}{h}
$$

- **Holomorphic example:** $f(z)=z^2$ gives one consistent derivative, $f'(z)=2z$.
- **Non-holomorphic example:** $f(z)=\lvert z \rvert^2=z\overline{z}$ depends on both $z$ and $\overline{z}$, so the ordinary complex derivative is not valid for optimization.

Most ML losses are real-valued and behave like the second case, which is the root of the problem we will see next.

### Direction Visual

This compares two tiny derivative estimates at the same point $z$:

- blue uses $h=\varepsilon$ (real-axis step)
- orange uses $h=i\varepsilon$ (imag-axis step)

If the arrows match, the derivative is direction-independent (holomorphic behavior).

<div id="holo-demo" style="max-width:700px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;">
  <canvas id="holo-canvas" width="640" height="340" style="width:100%;height:auto;display:block;"></canvas>
  <div style="margin-top:0.7rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="holo-x">Re(z)</label>
    <input id="holo-x" type="range" min="-2.0" max="2.0" step="0.1" value="1.0">
    <span id="holo-x-value">1.0</span>

    <label for="holo-y">Im(z)</label>
    <input id="holo-y" type="range" min="-2.0" max="2.0" step="0.1" value="0.8">
    <span id="holo-y-value">0.8</span>

    <label for="holo-eps">epsilon</label>
    <input id="holo-eps" type="range" min="0.01" max="0.25" step="0.01" value="0.08">
    <span id="holo-eps-value">0.08</span>
  </div>
  <div style="margin-top:0.5rem;font-size:0.92rem;line-height:1.35;">
    <div><strong>$f(z)=z^2$ gap:</strong> <span id="holo-gap-good">-</span></div>
    <div><strong>$f(z)=\lvert z \rvert^2$ gap:</strong> <span id="holo-gap-bad">-</span></div>
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('holo-canvas');
  const xInput = document.getElementById('holo-x');
  const yInput = document.getElementById('holo-y');
  const epsInput = document.getElementById('holo-eps');
  const xValue = document.getElementById('holo-x-value');
  const yValue = document.getElementById('holo-y-value');
  const epsValue = document.getElementById('holo-eps-value');
  const gapGood = document.getElementById('holo-gap-good');
  const gapBad = document.getElementById('holo-gap-bad');
  if (!canvas || !xInput || !yInput || !epsInput || !xValue || !yValue || !epsValue || !gapGood || !gapBad) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  function cAdd(a, b) { return { re: a.re + b.re, im: a.im + b.im }; }
  function cSub(a, b) { return { re: a.re - b.re, im: a.im - b.im }; }
  function cMul(a, b) { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
  function cDiv(a, b) {
    const d = b.re * b.re + b.im * b.im;
    return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
  }
  function cAbs(a) { return Math.hypot(a.re, a.im); }

  function fGood(z) {
    return cMul(z, z);
  }

  function fBad(z) {
    const r2 = z.re * z.re + z.im * z.im;
    return { re: r2, im: 0 };
  }

  function estimateDerivative(f, z, h) {
    return cDiv(cSub(f(cAdd(z, h)), f(z)), h);
  }

  function drawArrow(cx, cy, v, scale, color) {
    const ex = cx + v.re * scale;
    const ey = cy - v.im * scale;
    const ang = Math.atan2(ey - cy, ex - cx);
    const head = 8;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - head * Math.cos(ang - Math.PI / 6), ey - head * Math.sin(ang - Math.PI / 6));
    ctx.lineTo(ex - head * Math.cos(ang + Math.PI / 6), ey - head * Math.sin(ang + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function drawPanel(x0, y0, w, h, title, dReal, dImag, gap) {
    const cx = x0 + w / 2;
    const cy = y0 + h / 2 + 20;

    ctx.strokeStyle = '#3f4a56';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, w, h);

    ctx.strokeStyle = 'rgba(185,195,205,0.3)';
    ctx.beginPath();
    ctx.moveTo(x0 + 10, cy);
    ctx.lineTo(x0 + w - 10, cy);
    ctx.moveTo(cx, y0 + 28);
    ctx.lineTo(cx, y0 + h - 10);
    ctx.stroke();

    const maxMag = Math.max(0.4, cAbs(dReal), cAbs(dImag));
    const scale = Math.min(56, (w * 0.36) / maxMag);

    drawArrow(cx, cy, dReal, scale, '#4db0ff');
    drawArrow(cx, cy, dImag, scale, '#ffb74d');

    ctx.fillStyle = '#d9dee5';
    ctx.font = '13px sans-serif';
    ctx.fillText(title, x0 + 12, y0 + 18);
    ctx.fillStyle = '#4db0ff';
    ctx.fillText('h = epsilon (real step)', x0 + 12, y0 + h - 30);
    ctx.fillStyle = '#ffb74d';
    ctx.fillText('h = i epsilon (imag step)', x0 + 12, y0 + h - 12);

    ctx.fillStyle = '#d9dee5';
    ctx.fillText('gap = ' + gap.toExponential(2), x0 + 12, y0 + 36);
  }

  function render() {
    const z = { re: Number(xInput.value), im: Number(yInput.value) };
    const eps = Number(epsInput.value);
    const hReal = { re: eps, im: 0 };
    const hImag = { re: 0, im: eps };

    const goodReal = estimateDerivative(fGood, z, hReal);
    const goodImag = estimateDerivative(fGood, z, hImag);
    const badReal = estimateDerivative(fBad, z, hReal);
    const badImag = estimateDerivative(fBad, z, hImag);

    const goodGap = cAbs(cSub(goodReal, goodImag));
    const badGap = cAbs(cSub(badReal, badImag));

    xValue.textContent = z.re.toFixed(1);
    yValue.textContent = z.im.toFixed(1);
    epsValue.textContent = eps.toFixed(2);
    gapGood.textContent = goodGap.toExponential(3);
    gapBad.textContent = badGap.toExponential(3);

    ctx.fillStyle = '#10161d';
    ctx.fillRect(0, 0, W, H);

    drawPanel(14, 16, 300, 308, 'Holomorphic: f(z) = z^2', goodReal, goodImag, goodGap);
    drawPanel(326, 16, 300, 308, 'Non-holomorphic: f(z) = |z|^2', badReal, badImag, badGap);
  }

  xInput.addEventListener('input', render);
  yInput.addEventListener('input', render);
  epsInput.addEventListener('input', render);
  render();
})();
</script>

### Why Complex Derivatives Are Tough in ML

ML losses are real-valued (like $L = \lvert w-a \rvert^2$), which means they are non-holomorphic. Ordinary complex calculus only gives us one derivative $f'(w)$, and it **does not even exist** for these losses. So we cannot blindly reuse real-number gradient descent on complex weights.

If you try to apply the ordinary complex derivative anyway, something bad happens: the imaginary-axis direction gets flipped. The real-axis move is fine, but the vertical move goes the wrong way, and loss goes **up** instead of down. The graph below shows this directly. Blue walks toward the target. Red drifts away because its imaginary step is flipped.

<div id="complex-wirtinger-landscape" style="max-width:520px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;">
  <canvas id="complex-wirtinger-canvas" width="480" height="320" style="width:100%;height:auto;display:block;border-radius:6px;"></canvas>
  <div style="margin-top:0.7rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="wire-u0">start Re(w)</label>
    <input id="wire-u0" type="range" min="-2.0" max="2.0" step="0.1" value="-1.2">
    <span id="wire-u0-value">-1.2</span>

    <label for="wire-v0">start Im(w)</label>
    <input id="wire-v0" type="range" min="-1.0" max="3.5" step="0.1" value="0.9">
    <span id="wire-v0-value">0.9</span>

    <label for="wire-lr">learning rate</label>
    <input id="wire-lr" type="range" min="0.05" max="0.55" step="0.01" value="0.22">
    <span id="wire-lr-value">0.22</span>

    <label for="wire-steps">steps</label>
    <input id="wire-steps" type="range" min="4" max="28" step="1" value="16">
    <span id="wire-steps-value">16</span>
  </div>
  <div style="margin-top:0.6rem;display:flex;justify-content:space-between;align-items:center;gap:0.5rem;flex-wrap:wrap;">
    <button id="wire-replay" type="button" style="padding:0.25rem 0.55rem;border:1px solid #666;border-radius:6px;background:transparent;color:inherit;cursor:pointer;">Replay</button>
    <div style="font-size:0.92rem;line-height:1.35;">
      <strong>Step:</strong> <span id="wire-step">0</span>
      <strong style="margin-left:0.6rem;">Loss (correct):</strong> <span id="wire-loss-c">-</span>
      <strong style="margin-left:0.6rem;">Loss (wrong):</strong> <span id="wire-loss-w">-</span>
    </div>
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('complex-wirtinger-canvas');
  const u0Input = document.getElementById('wire-u0');
  const v0Input = document.getElementById('wire-v0');
  const lrInput = document.getElementById('wire-lr');
  const stepsInput = document.getElementById('wire-steps');
  const u0Value = document.getElementById('wire-u0-value');
  const v0Value = document.getElementById('wire-v0-value');
  const lrValue = document.getElementById('wire-lr-value');
  const stepsValue = document.getElementById('wire-steps-value');
  const replayButton = document.getElementById('wire-replay');
  const stepLabel = document.getElementById('wire-step');
  const lossCLabel = document.getElementById('wire-loss-c');
  const lossWLabel = document.getElementById('wire-loss-w');

  if (!canvas || !u0Input || !v0Input || !lrInput || !stepsInput || !u0Value || !v0Value || !lrValue || !stepsValue || !replayButton || !stepLabel || !lossCLabel || !lossWLabel) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const xMin = -2.4;
  const xMax = 2.4;
  const yMin = -1.2;
  const yMax = 3.8;
  const target = { re: 1.0, im: 2.0 };

  let startTime = performance.now();

  function toPxX(x) {
    return ((x - xMin) / (xMax - xMin)) * W;
  }

  function toPxY(y) {
    return H - ((y - yMin) / (yMax - yMin)) * H;
  }

  function loss(p) {
    const dx = p.re - target.re;
    const dy = p.im - target.im;
    return dx * dx + dy * dy;
  }

  function stepCorrect(w, lr) {
    return {
      re: w.re - lr * (w.re - target.re),
      im: w.im - lr * (w.im - target.im)
    };
  }

  function stepWrong(w, lr) {
    return {
      re: w.re - lr * (w.re - target.re),
      im: w.im + lr * (w.im - target.im)
    };
  }

  function trajectory(start, lr, steps, updater) {
    const pts = [start];
    for (let i = 0; i < steps; i += 1) {
      pts.push(updater(pts[pts.length - 1], lr));
    }
    return pts;
  }

  function drawBackground() {
    ctx.fillStyle = '#10161d';
    ctx.fillRect(0, 0, W, H);

    const sx = W / (xMax - xMin);
    const sy = H / (yMax - yMin);
    const levels = [0.2, 0.5, 1.0, 1.8, 3.0, 4.8, 7.0];

    ctx.strokeStyle = 'rgba(200,220,240,0.18)';
    ctx.lineWidth = 1;
    for (let i = 0; i < levels.length; i += 1) {
      const r = Math.sqrt(levels[i]);
      ctx.beginPath();
      ctx.ellipse(toPxX(target.re), toPxY(target.im), r * sx, r * sy, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function drawAxes() {
    const y0 = toPxY(0);
    const x0 = toPxX(0);
    ctx.strokeStyle = 'rgba(210,210,210,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(W, y0);
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, H);
    ctx.stroke();
  }

  function drawPoint(p, color, size) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(toPxX(p.re), toPxY(p.im), size, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawPath(points, color, width, upto) {
    if (upto < 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(toPxX(points[0].re), toPxY(points[0].im));
    for (let i = 1; i <= upto; i += 1) {
      ctx.lineTo(toPxX(points[i].re), toPxY(points[i].im));
    }
    ctx.stroke();
  }

  function drawLegend() {
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#4db0ff';
    ctx.fillText('correct descent direction', 14, 24);
    ctx.fillStyle = '#ff6b6b';
    ctx.fillText('naive complex derivative (flipped)', 14, 42);
  }

  function draw() {
    const start = { re: Number(u0Input.value), im: Number(v0Input.value) };
    const lr = Number(lrInput.value);
    const steps = Number(stepsInput.value);
    const good = trajectory(start, lr, steps, stepCorrect);
    const bad = trajectory(start, lr, steps, stepWrong);

    const elapsed = (performance.now() - startTime) / 1000;
    const stepFloat = Math.min(steps, elapsed * 4.0);
    const k = Math.floor(stepFloat);

    u0Value.textContent = start.re.toFixed(1);
    v0Value.textContent = start.im.toFixed(1);
    lrValue.textContent = lr.toFixed(2);
    stepsValue.textContent = String(steps);
    stepLabel.textContent = String(k);
    lossCLabel.textContent = loss(good[k]).toFixed(4);
    lossWLabel.textContent = loss(bad[k]).toFixed(4);

    drawBackground();
    drawAxes();

    drawPath(good, '#4db0ff', 3, k);
    drawPath(bad, '#ff6b6b', 3, k);

    drawPoint(target, '#f5f5f5', 5);
    drawPoint(start, '#d1d5db', 4);
    drawPoint(good[k], '#4db0ff', 4);
    drawPoint(bad[k], '#ff6b6b', 4);

    drawLegend();

    ctx.fillStyle = '#f5f5f5';
    ctx.font = '12px sans-serif';
    ctx.fillText('target a', toPxX(target.re) + 8, toPxY(target.im) - 8);

    if (k < steps) {
      requestAnimationFrame(draw);
    }
  }

  function restart() {
    startTime = performance.now();
    requestAnimationFrame(draw);
  }

  u0Input.addEventListener('input', restart);
  v0Input.addEventListener('input', restart);
  lrInput.addEventListener('input', restart);
  stepsInput.addEventListener('input', restart);
  replayButton.addEventListener('click', restart);
  restart();
})();
</script>

## Wirtinger Calculus has Entered the Fight

So if most ML loss functions are non-holomorphic, are we out of luck?

No. Wirtinger calculus lets us treat a real-valued complex loss in a way that is fully consistent with real gradient descent.

Thankfully some crazy guy named Wilhelm Wirtinger introduced this calculus back in 1927. ([Wirtinger Derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives).)

### Explaining Wirtinger

This smart math guy decided lets treat the complex numbers as two independent variables. A real-valued loss depends on **both** $z$ and $\overline{z}$. Since $z = u + iv$, that means the loss really just depends on two real numbers: $u$ and $v$. Two knobs to turn.

Because we have two knobs, we need two partial derivatives to describe how the loss changes. Wirtinger gives us exactly that:

$$
\frac{\partial L}{\partial z}
=
\frac{1}{2}\left(\frac{\partial L}{\partial u} - i\frac{\partial L}{\partial v}\right),
\qquad
\frac{\partial L}{\partial z^*}
=
\frac{1}{2}\left(\frac{\partial L}{\partial u} + i\frac{\partial L}{\partial v}\right)
$$

These are just the two real partial derivatives ($\partial L/\partial u$ and $\partial L/\partial v$) repackaged into complex form. One combines them with $-i$, the other with $+i$. That sign difference is the whole story.

Neither derivative is optional. They are the two partial derivatives of the calculus, just as $\partial L/\partial u$ and $\partial L/\partial v$ are both needed in real calculus.

For gradient descent on a real-valued loss, the update rule is:

$$
z_{t+1} = z_t - \eta \frac{\partial L}{\partial z^*}
$$

That is the only formula you need to remember. It looks like regular gradient descent but uses the conjugate derivative $\partial L/\partial z^{\ast}$.

If you unpack it, this is exactly the same as updating the real and imaginary parts separately:

$$
u_{t+1} = u_t - \frac{\eta}{2}\frac{\partial L}{\partial u},
\qquad
v_{t+1} = v_t - \frac{\eta}{2}\frac{\partial L}{\partial v}
$$

The factor of $1/2$ is the price of using the compact complex notation instead of writing out the two real updates.

If you want to go deeper, [Kreutz-Delgado's paper](https://arxiv.org/abs/0906.4835) walks through the full calculus with worked examples, and [Wikipedia's Wirtinger derivatives page](https://en.wikipedia.org/wiki/Wirtinger_derivatives) covers the formal math side.

#### Why the Wirtinger update uses $\partial L/\partial z^{\ast}$

So why $\partial L/\partial z^{\ast}$ and not $\partial L/\partial z$?

Simple. The conjugate derivative $\partial L/\partial z^{\ast}$ combines the real and imaginary partials with **+i**. That is exactly how we pack the two real updates into one complex number. The other derivative $\partial L/\partial z$ uses **-i**, which flips the imaginary step backward.

That is why the update rule uses $\partial L/\partial z^{\ast}$. Not by convention, but because it is the only one that actually descends.

**A concrete example.** Take $L = u^2 + v^2$ at the point $z = 1 + i$ (so $u=1, v=1$):

- Real gradient: $\partial L/\partial u = 2$, $\partial L/\partial v = 2$ → step goes toward $(-1, -1)$
- $\partial L/\partial z^{\ast} = 1 + i$ → update step $-2\eta(1 + i)$ moves down-left (correct)
- $\partial L/\partial z = 1 - i$ → update step $-2\eta(1 - i)$ moves down-right (wrong, imaginary part flips sign)

Now let's see why this works in general.

Start with the split real-imag update. We know this is correct for a real-valued loss:

$$
u_{t+1} - u_t = -\frac{\eta}{2}\frac{\partial L}{\partial u},
\qquad
v_{t+1} - v_t = -\frac{\eta}{2}\frac{\partial L}{\partial v}
$$

Combine those into a single complex step:

$$
\Delta z_{\text{real split}}
=
-\frac{\eta}{2}\frac{\partial L}{\partial u}
\;-\;
i\,\frac{\eta}{2}\frac{\partial L}{\partial v}
$$

Now plug in the Wirtinger derivative $\partial L/\partial z^{\ast} = \frac{1}{2}\!\left(\frac{\partial L}{\partial u} + i\frac{\partial L}{\partial v}\right)$:

$$
-\eta\frac{\partial L}{\partial z^*}
=
-\frac{\eta}{2}\frac{\partial L}{\partial u}
\;-\;
i\,\frac{\eta}{2}\frac{\partial L}{\partial v}
$$

Identical. Same real part, same imaginary part, same signs. This update walks exactly where real gradient descent would walk.

**What if we used the wrong one?** The two derivatives differ only in the sign of the imaginary piece:

> **Correct** ($\partial L/\partial z^{\ast}$): $\;-\frac{\eta}{2}\frac{\partial L}{\partial u} \;\mathbf{-}\; i\frac{\eta}{2}\frac{\partial L}{\partial v}$
>
> **Wrong** ($\partial L/\partial z$): $\;-\frac{\eta}{2}\frac{\partial L}{\partial u} \;\mathbf{+}\; i\frac{\eta}{2}\frac{\partial L}{\partial v}$

Same real part, opposite imaginary part. The wrong derivative flips the vertical move. The parameter goes in the opposite direction from what real gradient descent wants.

Both derivatives are part of the calculus, not an either-or menu. For real-valued losses the descent step uses $\partial L/\partial z^{\ast}$ because its algebra lines up with the real gradient on $(u, v)$.

**Bottom line:** $\partial L/\partial z^{\ast}$ is the one whose algebra matches real gradient descent. That is why the update rule uses it.

<div id="wirtinger-chains-demo" style="max-width:780px;margin:1.2rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;font-size:0.82rem;line-height:1.5;font-variant-numeric:tabular-nums;">
    <div>
      <div style="font-weight:700;color:#4db0ff;margin-bottom:0.3rem;">Correct (∂L/∂z*)</div>
      <div style="font-size:0.72rem;color:#999;margin-bottom:0.4rem;">Δv = −η · ∂L/∂v</div>
      <div id="wchain-good" style="max-height:420px;overflow-y:auto;scroll-behavior:smooth;"></div>
    </div>
    <div>
      <div style="font-weight:700;color:#ff6b6b;margin-bottom:0.3rem;">Wrong (∂L/∂z)</div>
      <div style="font-size:0.72rem;color:#999;margin-bottom:0.4rem;">Δv = +η · ∂L/∂v (flipped!)</div>
      <div id="wchain-bad" style="max-height:420px;overflow-y:auto;scroll-behavior:smooth;"></div>
    </div>
  </div>
  <div style="display:flex;gap:0.8rem;margin-top:0.7rem;flex-wrap:wrap;align-items:center;">
    <label style="font-size:0.85rem;">learning rate
      <input id="wchain-lr" type="range" min="0.02" max="0.40" step="0.01" value="0.15" style="width:110px;vertical-align:middle;">
      <span id="wchain-lr-val" style="font-size:0.85rem;min-width:2.4em;display:inline-block;">0.15</span>
    </label>
    <label style="font-size:0.85rem;">steps
      <input id="wchain-steps" type="range" min="2" max="8" step="1" value="5" style="width:90px;vertical-align:middle;">
      <span id="wchain-steps-val" style="font-size:0.85rem;">5</span>
    </label>
    <button id="wchain-replay" type="button" style="padding:0.2rem 0.5rem;border:1px solid #666;border-radius:6px;background:transparent;color:inherit;cursor:pointer;font-size:0.85rem;">Replay</button>
  </div>
</div>

<script>
(() => {
  const lrInput = document.getElementById('wchain-lr');
  const stepsInput = document.getElementById('wchain-steps');
  const lrVal = document.getElementById('wchain-lr-val');
  const stepsVal = document.getElementById('wchain-steps-val');
  const replayBtn = document.getElementById('wchain-replay');
  const goodContainer = document.getElementById('wchain-good');
  const badContainer = document.getElementById('wchain-bad');
  if (!lrInput || !stepsInput || !replayBtn || !goodContainer || !badContainer) return;

  const startU = 1.0, startV = 1.0;

  function buildChain(lr, steps, sign) {
    const chain = [];
    let u = startU, v = startV;
    for (let i = 0; i <= steps; i++) {
      const dLdu = 2 * u, dLdv = 2 * v;
      const du = -lr * dLdu;
      const dv = sign * lr * dLdv;
      chain.push({ step: i, u, v, dLdu, dLdv, du, dv, L: u * u + v * v });
      u += du;
      v += dv;
    }
    return chain;
  }

  function fmt(n) { return n.toFixed(3); }
  function fmtL(n) { return n.toFixed(4); }

  function createStepBlock(r, color, sign) {
    const block = document.createElement('div');
    block.style.cssText = 'margin-bottom:0.6rem;padding:0.5rem;border-radius:6px;opacity:0;transition:opacity 0.4s;';
    block.style.background = color === 'good' ? 'rgba(77,176,255,0.08)' : 'rgba(255,107,107,0.08)';
    block.style.border = '1px solid ' + (color === 'good' ? 'rgba(77,176,255,0.2)' : 'rgba(255,107,107,0.2)');

    const dvColor = color === 'good' ? '#4db0ff' : '#ff6b6b';
    const dvSign = sign < 0 ? '-' : '+';

    block.innerHTML = `
      <div style="font-weight:600;color:#ccc;margin-bottom:0.3rem;">Step ${r.step}</div>
      <div class="wchain-line" style="opacity:0;transition:opacity 0.3s;">
        <span style="color:#aaa;">u =</span> <span style="color:#e0e0e0;">${fmt(r.u)}</span>,
        <span style="color:#aaa;">v =</span> <span style="color:#e0e0e0;">${fmt(r.v)}</span>
      </div>
      <div class="wchain-arrow" style="opacity:0;transition:opacity 0.3s;text-align:center;color:#666;margin:0.15rem 0;">↓</div>
      <div class="wchain-line" style="opacity:0;transition:opacity 0.3s;">
        <span style="color:#aaa;">∂L/∂u =</span> <span style="color:#e0e0e0;">${fmt(r.dLdu)}</span>,
        <span style="color:#aaa;">∂L/∂v =</span> <span style="color:#e0e0e0;">${fmt(r.dLdv)}</span>
      </div>
      <div class="wchain-arrow" style="opacity:0;transition:opacity 0.3s;text-align:center;color:#666;margin:0.15rem 0;">↓</div>
      <div class="wchain-line" style="opacity:0;transition:opacity 0.3s;">
        <span style="color:#aaa;">Δu =</span> <span style="color:#e0e0e0;">${fmt(r.du)}</span>,
        <span style="color:#aaa;">Δv =</span> <span style="color:${dvColor};font-weight:700;">${dvSign}${fmt(Math.abs(r.dv))}</span>
      </div>
      <div class="wchain-arrow" style="opacity:0;transition:opacity 0.3s;text-align:center;color:#666;margin:0.15rem 0;">↓</div>
      <div class="wchain-line" style="opacity:0;transition:opacity 0.3s;">
        <span style="color:#aaa;">L =</span> <span style="color:#e0e0e0;">${fmtL(r.L)}</span>
      </div>
    `;
    return block;
  }

  function revealStep(block) {
    block.style.opacity = '1';
    const lines = block.querySelectorAll('.wchain-line, .wchain-arrow');
    lines.forEach((el, i) => {
      setTimeout(() => { el.style.opacity = '1'; }, i * 120);
    });
  }

  function fadeOldSteps(container) {
    const blocks = container.querySelectorAll(':scope > div');
    blocks.forEach((b, i) => {
      if (i < blocks.length - 1) {
        b.style.opacity = '0.35';
      }
    });
  }

  let animTimer = null;

  function animate() {
    const lr = Number(lrInput.value);
    const steps = Number(stepsInput.value);
    const goodChain = buildChain(lr, steps, -1);
    const badChain = buildChain(lr, steps, +1);

    lrVal.textContent = lr.toFixed(2);
    stepsVal.textContent = String(steps);

    goodContainer.innerHTML = '';
    badContainer.innerHTML = '';

    let currentStep = 0;

    function showNext() {
      if (currentStep > steps) {
        if (animTimer) clearTimeout(animTimer);
        return;
      }

      const gBlock = createStepBlock(goodChain[currentStep], 'good', -1);
      const bBlock = createStepBlock(badChain[currentStep], 'bad', +1);

      goodContainer.appendChild(gBlock);
      badContainer.appendChild(bBlock);

      fadeOldSteps(goodContainer);
      fadeOldSteps(badContainer);

      setTimeout(() => {
        revealStep(gBlock);
        revealStep(bBlock);
        goodContainer.scrollTop = goodContainer.scrollHeight;
        badContainer.scrollTop = badContainer.scrollHeight;
      }, 50);

      currentStep++;
      animTimer = setTimeout(showNext, 1200);
    }

    showNext();
  }

  function restart() {
    if (animTimer) clearTimeout(animTimer);
    animate();
  }

  lrInput.addEventListener('input', restart);
  stepsInput.addEventListener('input', restart);
  replayBtn.addEventListener('click', restart);
  restart();
})();
</script>

### Wirtinger Solves the Derivative Visual

This visual shows the toy loss $L(w) = \lvert w-a \rvert^2$ with a rotating target. Pick one of the two derivatives using the buttons and watch what happens.

- **Wirtinger (correct):** use $$\frac{\partial L}{\partial w^*}$$, so the step $$w - \eta\frac{\partial L}{\partial w^*}$$ matches the real-imag gradient. $w$ tracks the target and loss falls.
- **Naive complex derivative:** use $$\frac{\partial L}{\partial w}$$, which is the conjugate. It flips the imaginary step, so $w$ drifts away and loss rises.

Both derivatives are defined because the calculus needs both to describe a non-holomorphic function. The visual shows which one corresponds to real gradient descent.

### Why It Captures Rotation Better

Here is the part that connects back to the rotation theme of this whole article.

A complex weight $w$ carries both magnitude and angle. When the target $a$ sits at some angle, the error vector $w - a$ points in a specific direction. The Wirtinger update $\frac{\partial L}{\partial w^{\ast}} = w - a$ preserves that direction, so each step moves $w$ along the true line toward $a$.

The naive derivative $\frac{\partial L}{\partial w} = \overline{(w-a)}$ conjugates the error, which mirrors the angle across the real axis. That mirror flip is exactly the rotation bug: the step points in a direction that does not match the real geometry of the loss surface.

So Wirtinger derivatives "capture rotation" because they respect the complex structure of the error instead of silently flipping part of it.

For the toy loss used in the code,

$$
L(w)=\lvert w-a \rvert^2
$$

we get

$$
\frac{\partial L}{\partial u}=2\,\Re(w-a),
\qquad
\frac{\partial L}{\partial v}=2\,\Im(w-a)
$$

so

$$
\frac{\partial L}{\partial w^*}=w-a,
\qquad
\frac{\partial L}{\partial w}=\overline{(w-a)}
$$

### Rotating Target Visual

This visual shows the difference directly. The target $a$ rotates around the origin, so the model has to track a moving angle. The blue path uses $\frac{\partial L}{\partial w^{\ast}}$ and follows the target. The red path uses $\frac{\partial L}{\partial w}$ and drifts away because the imaginary update is mirrored.

<div id="wirtinger-rotation-demo" style="max-width:560px;margin:1rem auto;padding:0.9rem;border:1px solid #444;border-radius:10px;">
  <canvas id="wirtinger-rotation-canvas" width="500" height="360" style="width:100%;height:auto;display:block;border-radius:6px;"></canvas>
  <div style="margin-top:0.7rem;font-size:0.9rem;display:grid;grid-template-columns:auto 1fr auto;gap:0.35rem;align-items:center;">
    <label for="wir-rot-speed">target speed</label>
    <input id="wir-rot-speed" type="range" min="0" max="3" step="0.1" value="1.0">
    <span id="wir-rot-speed-value">1.0</span>

    <label for="wir-rot-lr">learning rate</label>
    <input id="wir-rot-lr" type="range" min="0.02" max="0.20" step="0.01" value="0.06">
    <span id="wir-rot-lr-value">0.06</span>

    <label for="wir-rot-radius">target radius</label>
    <input id="wir-rot-radius" type="range" min="0.5" max="2.0" step="0.1" value="1.2">
    <span id="wir-rot-radius-value">1.2</span>
  </div>
  <div style="margin-top:0.6rem;font-size:0.92rem;line-height:1.4;">
    <div><span style="color:#4db0ff;">&#9632;</span> Wirtinger: tracks the rotating target</div>
    <div><span style="color:#ff6b6b;">&#9632;</span> Naive: drifts away (imag axis flipped), clamped to plot</div>
    <div><span style="color:#f5f5f5;">&#9632;</span> target a (rotating)</div>
    <div style="margin-top:0.35rem;">
      <strong>Loss (Wirtinger):</strong> <span id="wir-rot-loss-c">-</span>
      <strong style="margin-left:0.6rem;">Loss (naive):</strong> <span id="wir-rot-loss-w">-</span>
    </div>
  </div>
</div>

<script>
(() => {
  const canvas = document.getElementById('wirtinger-rotation-canvas');
  const speedInput = document.getElementById('wir-rot-speed');
  const lrInput = document.getElementById('wir-rot-lr');
  const radiusInput = document.getElementById('wir-rot-radius');
  const speedValue = document.getElementById('wir-rot-speed-value');
  const lrValue = document.getElementById('wir-rot-lr-value');
  const radiusValue = document.getElementById('wir-rot-radius-value');

  if (!canvas || !speedInput || !lrInput || !radiusInput || !speedValue || !lrValue || !radiusValue) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const xMin = -2.6;
  const xMax = 2.6;
  const yMin = -2.2;
  const yMax = 2.2;
  let theta = 0;
  let wCorrect = { re: -1.6, im: -1.2 };
  let wWrong = { re: -1.6, im: -1.2 };
  let trailCorrect = [];
  let trailWrong = [];
  let lossC = 0;
  let lossW = 0;
  const maxTrail = 180;

  const padRe = 0.15 * (xMax - xMin);
  const padIm = 0.15 * (yMax - yMin);

  function toPxX(x) { return ((x - xMin) / (xMax - xMin)) * W; }
  function toPxY(y) { return H - ((y - yMin) / (yMax - yMin)) * H; }

  function clampPoint(p) {
    return {
      re: Math.max(xMin + padRe, Math.min(xMax - padRe, p.re)),
      im: Math.max(yMin + padIm, Math.min(yMax - padIm, p.im))
    };
  }

  function restart() {
    theta = 0;
    wCorrect = { re: -1.6, im: -1.2 };
    wWrong = { re: -1.6, im: -1.2 };
    trailCorrect = [];
    trailWrong = [];
  }

  function loss(p, target) {
    const dx = p.re - target.re;
    const dy = p.im - target.im;
    return dx * dx + dy * dy;
  }

  function stepCorrect(w, target, lr) {
    return {
      re: w.re - lr * (w.re - target.re),
      im: w.im - lr * (w.im - target.im)
    };
  }

  function stepWrong(w, target, lr) {
    return {
      re: w.re - lr * (w.re - target.re),
      im: w.im + lr * (w.im - target.im)
    };
  }

  function drawBackground() {
    ctx.fillStyle = '#10161d';
    ctx.fillRect(0, 0, W, H);

    const sx = W / (xMax - xMin);
    const sy = H / (yMax - yMin);
    const levels = [0.5, 1.0, 2.0, 3.5];

    ctx.strokeStyle = 'rgba(200,220,240,0.12)';
    ctx.lineWidth = 1;
    for (let i = 0; i < levels.length; i += 1) {
      const r = Math.sqrt(levels[i]);
      ctx.beginPath();
      ctx.ellipse(toPxX(0), toPxY(0), r * sx, r * sy, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  function drawAxes() {
    const y0 = toPxY(0);
    const x0 = toPxX(0);
    ctx.strokeStyle = 'rgba(210,210,210,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y0);
    ctx.lineTo(W, y0);
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, H);
    ctx.stroke();
  }

  function drawPoint(p, color, size) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(toPxX(p.re), toPxY(p.im), size, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawTrail(trail, color) {
    if (trail.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toPxX(trail[0].re), toPxY(trail[0].im));
    for (let i = 1; i < trail.length; i += 1) {
      ctx.lineTo(toPxX(trail[i].re), toPxY(trail[i].im));
    }
    ctx.stroke();
  }

  function frame() {
    const speed = Number(speedInput.value);
    const lr = Number(lrInput.value);
    const radius = Number(radiusInput.value);

    speedValue.textContent = speed.toFixed(1);
    lrValue.textContent = lr.toFixed(2);
    radiusValue.textContent = radius.toFixed(1);

    const target = {
      re: radius * Math.cos(theta),
      im: radius * Math.sin(theta)
    };

    wCorrect = stepCorrect(wCorrect, target, lr);
    wWrong = stepWrong(wWrong, target, lr);

    lossC = loss(wCorrect, target);
    lossW = loss(wWrong, target);

    trailCorrect.push({ re: wCorrect.re, im: wCorrect.im });
    trailWrong.push(clampPoint(wWrong));
    if (trailCorrect.length > maxTrail) trailCorrect.shift();
    if (trailWrong.length > maxTrail) trailWrong.shift();

    theta += speed * 0.03;

    drawBackground();
    drawAxes();

    drawTrail(trailWrong, 'rgba(255,107,107,0.55)');
    drawTrail(trailCorrect, 'rgba(77,176,255,0.7)');

    const wrongDraw = clampPoint(wWrong);

    drawPoint(target, '#f5f5f5', 5);
    drawPoint(wCorrect, '#4db0ff', 4);
    drawPoint(wrongDraw, '#ff6b6b', 4);

    const lossCLabel = document.getElementById('wir-rot-loss-c');
    const lossWLabel = document.getElementById('wir-rot-loss-w');
    if (lossCLabel) lossCLabel.textContent = lossC.toFixed(3);
    if (lossWLabel) lossWLabel.textContent = lossW.toFixed(3);

    ctx.fillStyle = '#f5f5f5';
    ctx.font = '12px sans-serif';
    ctx.fillText('target a', toPxX(target.re) + 8, toPxY(target.im) - 8);
    ctx.fillStyle = '#4db0ff';
    ctx.fillText('w (Wirtinger)', toPxX(wCorrect.re) + 8, toPxY(wCorrect.im) - 8);
    ctx.fillStyle = '#ff6b6b';
    ctx.fillText('w (naive)', toPxX(wrongDraw.re) + 8, toPxY(wrongDraw.im) - 8);

    requestAnimationFrame(frame);
  }

  frame();
})();
</script>
