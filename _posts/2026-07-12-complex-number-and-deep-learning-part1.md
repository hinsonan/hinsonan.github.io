---
layout: post
title: "Complex Numbers and Deep Learning Part 1"
date: 2026-07-12
categories: ML
---

No one knows what an imaginary number is. It's like asking ML people how this algorithm is meant to work outside of a notebook. That reality just does not exist. Imaginary numbers are very powerful for certain areas of engineering. Deep learning algorithms can also use imaginary numbers but often do not due to limitations of back propagation. There are ways to get around this and you can keep the natural relationship between real numbers and imaginary. Keeping this relationship together can have the model learn more efficiently and generalize better. Before we get into all that we need to understand what an imaginary number is and what a complex number is.

## Complex Numbers

Imaginary numbers are a bad name. Imaginary makes it seem like it's not real but it is. This is in some ways hard to grasp like negative numbers were hard to grasp for many in mathematics. Which makes sense because if you have 3 apples and you take 4 apples away you now have -1 apples. How in the world can you have -1 apples? It sounds absurd but that doesn't mean negative numbers are not real. In banking it makes a lot of sense. You are in the hole and have negative dollars. You gotta get out of that hole.

Wikipedia defines an imaginary number as: "the product of a real number and the imaginary unit $i$, which is defined by its property $i^2 = -1$."

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

### Polar Form (Intuition First)

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

Here is a tiny animation based on the idea of `plot_rotation_vectors(x=1.0, y=1.0, angle_deg=45.0)`.
It draws the original vector and its rotated version.

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
