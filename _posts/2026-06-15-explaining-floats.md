---
layout: post
title: "Explaining Floats: TF32, BF16, FP8, FP4, Huh?"
date: 2026-06-13
categories: ML
---

There are too many floats now and no one knows what they are. Everyone knows about 64- and 32-bit floats but over the past 10 years all these "new" types have emerged. You start mentioning brain floats to someone and you have lost the audience. If you work in ML or even if you primarily work on serving models for inference you need to understand these data types. Grab hold of your exponents and let's see where the mantissa takes us.

# Classic IEEE 754

We must understand what a floating point is and how CPUs have been using them. In 1985 [IEEE](https://en.wikipedia.org/wiki/IEEE_754) established the standard for floating point numbers. All floats comprise three things

1) Sign bit

2) Exponent

3) Mantissa

<div class="mermaid">
flowchart TD
    A[Floating Point] --> B[Sign Bit]
    A --> C[Exponent Bits]
    A --> D[Mantissa Bits]
    B --> B1["1 bit <br/>0=Positive, 1=Negative"]
    C --> C1[Biased exponent for range]
    D --> D1[Fractional precision]
</div>

The sign is 1 bit that is either 0 (positive) or 1 (negative)

The exponent controls the range (how small or large) of the number. The more exponents the wider the range.

The mantissa controls the precision of the number. It stores the fractional part. More mantissa bits means you will have a more precise value.

## Examples of 64 and 32 bit numbers

Here are what some examples would look like. These are the types of computations that CPUs have been doing for a very long time.

### The IEEE 754 formula

IEEE 754 standard states that the value of a binary32 (FP32) number is:

```
value = (-1)^sign × 2^(exponent - bias) × 1.mantissa
```

Where:

- **sign** = bit 31 (0 = positive, 1 = negative)
- **exponent** = the 8-bit unsigned integer stored in bits 30-23
- **bias** = 127 for FP32, 1023 for FP64
- **mantissa** = the 23 fraction bits (bits 22-0), with an **implicit leading 1** for normal numbers

This formula is like scientific notation in binary: `(-1)^sign × 1.m × 2^(E-bias)`.

### FP32 bit layout examples

**0.15625 in FP32**

`0 | 01111100 | 01000000000000000000000`

<div style="overflow-x:auto;margin:0.5rem 0;">
<table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
  <tr>
    <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.85rem;">sign</th>
    <th colspan="8" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.85rem;">exponent (8 bits)</th>
    <th colspan="23" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.85rem;">fraction (23 bits)</th>
  </tr>
  <tr>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#3b82f6;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
  </tr>
  <tr>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">31</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">30</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">23</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">22</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">0</td>
  </tr>
</table>
</div>

**= 0.15625**

Applying the formula:

- `sign = 0` → `(-1)^0 = +1`
- `exponent = 01111100₂ = 124` → `2^(124 - 127) = 2^(-3) = 1/8`
- `mantissa = .0100...0₂ = 1/4` → `1 + 1/4 = 1.25`

`(+1) × 1.25 × 1/8 = 0.15625`

**-2.5 in FP32**

`1 | 10000000 | 01000000000000000000000`

<div style="overflow-x:auto;margin:0.5rem 0;">
<table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
  <tr>
    <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.85rem;">sign</th>
    <th colspan="8" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.85rem;">exponent (8 bits)</th>
    <th colspan="23" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.85rem;">fraction (23 bits)</th>
  </tr>
  <tr>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#3b82f6;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:18px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">0</td>
  </tr>
  <tr>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">31</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">30</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">23</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">22</td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">0</td>
  </tr>
</table>
</div>

**= -2.5**

Applying the formula:

- `sign = 1` → `(-1)^1 = -1`
- `exponent = 10000000₂ = 128` → `2^(128 - 127) = 2^1 = 2`
- `mantissa = .0100...0₂ = 1/4` → `1 + 1/4 = 1.25`

`(-1) × 1.25 × 2 = -2.5`

### FP64 examples

For FP64, the formula is the same but with **bias = 1023**:

```
value = (-1)^sign x 2^(exponent - 1023) x 1.mantissa
```

**0.15625 in FP64**

`0 | 01111111100 | 010000...000000`

<div style="overflow-x:auto;margin:0.5rem 0;">
<table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
  <tr>
    <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.75rem;">sign</th>
    <th colspan="11" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.75rem;">exponent (11 bits)</th>
    <th colspan="52" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.75rem;">fraction (52 bits)</th>
  </tr>
  <tr>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#3b82f6;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
  </tr>
  <tr>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">63</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">62</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">52</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">51</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">0</td>
  </tr>
</table>
</div>

**= 0.15625**

Applying the formula:

- `sign = 0` -> `(-1)^0 = +1`
- `exponent = 01111111100₂ = 1020` -> `2^(1020 - 1023) = 2^-3 = 1/8`
- `mantissa = .0100...0₂ = 1/4` -> `1 + 1/4 = 1.25`

`(+1) x 1.25 x 1/8 = 0.15625`

**-2.5 in FP64**

`1 | 10000000000 | 010000...000000`

<div style="overflow-x:auto;margin:0.5rem 0;">
<table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
  <tr>
    <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.75rem;">sign</th>
    <th colspan="11" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.75rem;">exponent (11 bits)</th>
    <th colspan="52" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.75rem;">fraction (52 bits)</th>
  </tr>
  <tr>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#3b82f6;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#22c55e;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">1</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
    <td style="border:1px solid #1f2937;width:10px;height:22px;text-align:center;font-weight:bold;font-size:0.65rem;background:#ef4444;color:#fff;padding:0;">0</td>
  </tr>
  <tr>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">63</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">62</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">52</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">51</td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;"></td>
    <td style="text-align:center;font-size:0.55rem;color:#6b7280;padding:1px 0;">0</td>
  </tr>
</table>
</div>

**= -2.5**

Applying the formula:

- `sign = 1` -> `(-1)^1 = -1`
- `exponent = 10000000000₂ = 1024` -> `2^(1024 - 1023) = 2^1 = 2`
- `mantissa = .0100...0₂ = 1/4` -> `1 + 1/4 = 1.25`

`(-1) x 1.25 x 2 = -2.5`

Now that we understand this a bit better we can move onto all these new types that have become so popular with GPUs and ML

# GPUs get all the floats

Let's break down these different types and when they released.

## Release Dates

| Type | Year | Hardware | Notes |
|------|------|----------|-------|
| FP64, FP32 | 1985 | IEEE 754 | the originals |
| FP16 | 2002 | GeForce FX | first GPU float16 (graphics only) |
| FP16 | 2016 | Pascal GP100 | real FP16 ML compute support |
| BF16 | 2017 | Google TPU v2 | ML focused 16-bit format |
| BF16 | 2020 | Ampere / Cooper Lake | widespread GPU/CPU adoption |
| TF32 | 2020 | Ampere A100 | FP32 speedup |
| FP8 | 2022 | Hopper H100 | 8-bit training |
| FP4 | 2024 | Blackwell B200 | 4-bit inference | 

## Floating Point Breakdowns

| Type | Sign | Exponent | Mantissa | Total Bits |
|------|------|----------|----------|------------|
| FP64 / double | 1 | 11 | 52 | 64 |
| FP32 / float | 1 | 8 | 23 | 32 |
| TF32 | 1 | 8 | 10 | 19 |
| BF16 | 1 | 8 | 7 | 16 |
| FP16 / half | 1 | 5 | 10 | 16 |
| FP8 (E4M3) | 1 | 4 | 3 | 8 |
| FP8 (E5M2) | 1 | 5 | 2 | 8 |
| FP4 | 1 | 2 | 1 | 4 |

## Tensor Float 32 (TF32)

Around 2017 Nvidia started making GPUs with Tensor Cores (Volta V100), and with Ampere (A100) in 2020 they added TF32 support. These are in total 19 bits and they have an accumulator that accumulates back to a float32. The loss of precision does not matter for training and the loss is pretty small. Faster training speeds is worth the precision.

```
Input:  FP32 (23-bit mantissa)
           ↓
Multiply: TF32 (10-bit mantissa)
           ↓
Accumulate: FP32 (23-bit mantissa)
```

The way this works is that the precision is dropped to 10 mantissa at multiply but when adding products together that is done at the full 23 bit mantissa. The GPU has specific tensor cores that are created and optimized for these calculations. Below is a more complete example.

<div style="font-family: monospace; margin: 16px 0; padding: 16px; background: #1a1a2e; border-radius: 8px; color: #fff; overflow-x: auto;">

<div style="font-size: 12px; color: #fbbf24; margin-bottom: 4px;">1) 2.5 and 4.5 enter as FP32 (32-bit fields on the wire)</div>
<table style="border-collapse: collapse; margin: 2px 0;">
<tr><td style="padding: 1px 4px; color: #888; font-size: 10px;">2.5</td></tr>
<tr><td style="padding: 0;"><table style="border-collapse: collapse;"><tr>
<td style="padding: 1px 3px; background: #3b82f6; color: #fff; font-size: 9px; border: 1px solid #2563eb; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">1</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">1</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #374151; color: #666; font-size: 9px; border: 1px solid #1f2937; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #374151; color: #666; font-size: 9px; border: 1px solid #1f2937; min-width: 22px; text-align: center;" colspan="10">...0s (20 more bits)</td>
</tr></table></td></tr>
</table>
<table style="border-collapse: collapse; margin: 2px 0;">
<tr><td style="padding: 1px 4px; color: #888; font-size: 10px;">4.5</td></tr>
<tr><td style="padding: 0;"><table style="border-collapse: collapse;"><tr>
<td style="padding: 1px 3px; background: #3b82f6; color: #fff; font-size: 9px; border: 1px solid #2563eb; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">1</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #22c55e; color: #fff; font-size: 9px; border: 1px solid #16a34a; min-width: 22px; text-align: center;">1</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">0</td>
<td style="padding: 1px 3px; background: #a855f7; color: #fff; font-size: 9px; border: 1px solid #9333ea; min-width: 22px; text-align: center;">1</td>
<td style="padding: 1px 3px; background: #374151; color: #666; font-size: 9px; border: 1px solid #1f2937; min-width: 22px; text-align: center;" colspan="10">...0s (20 more bits)</td>
</tr></table></td></tr>
</table>

<div style="font-size: 11px; color: #888; margin: 2px 0;"><span style="color: #3b82f6;">sign</span> <span style="color: #22c55e;">exponent</span> <span style="color: #a855f7;">kept (10-bit mantissa)</span> <span style="color: #374151;">ignored (13 bits)</span></div>

<div style="font-size: 12px; color: #fbbf24; margin: 10px 0 4px;">2) Multiply reads only the purple bits (first 10 mantissa bits)</div>

<div style="font-size: 12px; color: #fff; margin: 4px 0;">&nbsp;&nbsp;TF32 multiply: 2.5 × 4.5 = 11.25</div>

<div style="font-size: 12px; color: #fbbf24; margin: 10px 0 4px;">3) Each K-step feeds one product into the FP32 accumulator (hardware adder on die):</div>

<div style="margin: 8px 0;">
<table style="border-collapse: collapse; margin: 0 auto; background: #16213e;">
<tr style="border-bottom: 1px solid #0f3460;">
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">K-step</th>
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">Multiply</th>
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">Partial Product</th>
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">acc before</th>
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">acc after</th>
<th style="padding: 4px 10px; color: #888; font-size: 10px; border: 1px solid #0f3460;">Precision</th>
</tr>
<tr>
<td style="padding: 4px 10px; color: #aaa; font-size: 11px; border: 1px solid #0f3460; text-align: center;">0</td>
<td style="padding: 4px 10px; color: #fff; font-size: 11px; border: 1px solid #0f3460;">2.5×4.5</td>
<td style="padding: 4px 10px; background: #2563eb; color: #93c5fd; font-size: 11px; border: 1px solid #0f3460; text-align: center;">11.25</td>
<td style="padding: 4px 10px; background: #9333ea; color: #d8b4fe; font-size: 11px; border: 1px solid #0f3460; text-align: center;">0.0</td>
<td style="padding: 4px 10px; background: #dc2626; color: #fca5a5; font-size: 11px; border: 1px solid #0f3460; text-align: center;">11.25</td>
<td style="padding: 4px 6px; color: #22c55e; font-size: 10px; border: 1px solid #0f3460; text-align: center;">FP32</td>
</tr>
<tr>
<td style="padding: 4px 10px; color: #aaa; font-size: 11px; border: 1px solid #0f3460; text-align: center;">1</td>
<td style="padding: 4px 10px; color: #fff; font-size: 11px; border: 1px solid #0f3460;">3.1×2.2</td>
<td style="padding: 4px 10px; background: #2563eb; color: #93c5fd; font-size: 11px; border: 1px solid #0f3460; text-align: center;">6.82</td>
<td style="padding: 4px 10px; background: #9333ea; color: #d8b4fe; font-size: 11px; border: 1px solid #0f3460; text-align: center;">11.25</td>
<td style="padding: 4px 10px; background: #dc2626; color: #fca5a5; font-size: 11px; border: 1px solid #0f3460; text-align: center;">18.07</td>
<td style="padding: 4px 6px; color: #22c55e; font-size: 10px; border: 1px solid #0f3460; text-align: center;">FP32</td>
</tr>
<tr>
<td style="padding: 4px 10px; color: #aaa; font-size: 11px; border: 1px solid #0f3460; text-align: center;">2</td>
<td style="padding: 4px 10px; color: #fff; font-size: 11px; border: 1px solid #0f3460;">1.7×0.9</td>
<td style="padding: 4px 10px; background: #2563eb; color: #93c5fd; font-size: 11px; border: 1px solid #0f3460; text-align: center;">1.53</td>
<td style="padding: 4px 10px; background: #9333ea; color: #d8b4fe; font-size: 11px; border: 1px solid #0f3460; text-align: center;">18.07</td>
<td style="padding: 4px 10px; background: #dc2626; color: #fca5a5; font-size: 11px; border: 1px solid #0f3460; text-align: center;">19.60</td>
<td style="padding: 4px 6px; color: #22c55e; font-size: 10px; border: 1px solid #0f3460; text-align: center;">FP32</td>
</tr>
<tr>
<td style="padding: 4px 10px; color: #aaa; font-size: 11px; border: 1px solid #0f3460; text-align: center;">⋮</td>
<td style="padding: 4px 10px; color: #555; font-size: 11px; border: 1px solid #0f3460;">⋮</td>
<td style="padding: 4px 10px; color: #555; font-size: 11px; border: 1px solid #0f3460;">⋮</td>
<td style="padding: 4px 10px; color: #555; font-size: 11px; border: 1px solid #0f3460;">⋮</td>
<td style="padding: 4px 10px; color: #555; font-size: 11px; border: 1px solid #0f3460;">⋮</td>
<td style="padding: 4px 6px; color: #555; font-size: 10px; border: 1px solid #0f3460; text-align: center;"></td>
</tr>
<tr>
<td style="padding: 4px 10px; color: #aaa; font-size: 11px; border: 1px solid #0f3460; text-align: center;">15</td>
<td style="padding: 4px 10px; color: #fff; font-size: 11px; border: 1px solid #0f3460;">0.4×8.1</td>
<td style="padding: 4px 10px; background: #2563eb; color: #93c5fd; font-size: 11px; border: 1px solid #0f3460; text-align: center;">3.24</td>
<td style="padding: 4px 10px; background: #9333ea; color: #d8b4fe; font-size: 11px; border: 1px solid #0f3460; text-align: center;">...</td>
<td style="padding: 4px 10px; background: #dc2626; color: #fca5a5; font-size: 11px; border: 1px solid #0f3460; text-align: center;">final dot product</td>
<td style="padding: 4px 6px; color: #22c55e; font-size: 10px; border: 1px solid #0f3460; text-align: center;">FP32</td>
</tr>
</table>
</div>

</div>

Since the accumulator keeps full FP32 precision during addition you limit the precision loss. For scientific computing this would not be a good idea but for ML training this precision is rarely worth the cost for larger multi-billion param models.

## Brain Float 16 (BF16)

BF16 and FP16 are the exact same amount of bits so what gives? The range of the numbers for BF16 is greater than FP16 but with lesser precision. BF16 has 8 exponents and FP16 has 5.

The primary reason for BF16 was to make training easier by reducing the need to apply loss scaling when training large models (billions of params)

To protect FP16 from underflow and overflow during backprop you need to scale the values. The reason you have to protect against this is because for large models the logits can get very large and overflow the range of FP16. The other issue is that the vanishing gradient issue can pop up when training the large models. So you need to scale the loss which looks something like this

```python
loss_scaled = loss × S              # one scalar, cheap

# backprop runs: ALL gradients are now ×S in size
# FP16 can represent them because they're no longer tiny

# BEFORE optimizer.step():
for grad in model.parameters():
    grad /= S                       # done in FP32, exact
    
optimizer.step()                    # sees correct-magnitude gradients
```

Using Pytorch this is done by using `GradScaler`

```python
scaler = torch.amp.GradScaler("cuda")   # manages S automatically

for batch in dataloader:
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = model(batch)            # forward in FP16

    scaler.scale(loss).backward()      # loss × S, then backprop (grads are ×S)
    scaler.unscale_(optimizer)         # divides all grads by S in FP32
    scaler.step(optimizer)             # optimizer sees correct magnitudes
    scaler.update()                    # adjusts S up or down for next step
```

The problem with this is sometimes its very hard to find the right value to scale by. Many times you can solve overflow but not underflow. In order to solve this BF16 comes to the rescue.

BF16 has the same range as FP32 but with less precision. This is well worth the trade off since now we don't have to apply scaling and deal with these issues. Our gradients can flow more properly.

There are some instances where you may prefer the extra precision of FP16 such as after pre-training and supervised training. The weights are more stable and maybe if you want to do reinforcement learning you may want to switch to FP16 for extra precision. This is not always true and many people have used BF16 for RL tuning.

## Floating Point 8 (FP8) and Floating Point 4 (FP4)

FP8 comes in two versions

* **E4M3** (4 exponent, 3 mantissa)
* **E5M2** (5 exponent, 2 mantissa). 

Both are needed because activations and gradients have different requirements:

- **E4M3** (range ±448): More precision for weights and activations (forward pass).
- **E5M2** (range ±57344): More range for gradients (backward pass). Gradients span many orders of magnitude and need the wider range to avoid overflow.

<div style="overflow-x:auto;margin:1rem 0;">
  <div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;"><strong>FP8 E4M3</strong> (1 sign | 4 exponent | 3 mantissa): weights and activations, max value 448</div>
  <table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
    <tr>
      <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.85rem;">sign</th>
      <th colspan="4" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.85rem;">exponent (4 bits)</th>
      <th colspan="3" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.85rem;">mantissa (3 bits)</th>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#3b82f6;color:#fff;padding:0;">S</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
    </tr>
    <tr>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">7</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">6</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">3</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">2</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">0</td>
    </tr>
  </table>
</div>

<div style="overflow-x:auto;margin:1rem 0;">
  <div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;"><strong>FP8 E5M2</strong> (1 sign | 5 exponent | 2 mantissa): gradients, max value 57344</div>
  <table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
    <tr>
      <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.85rem;">sign</th>
      <th colspan="5" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.85rem;">exponent (5 bits)</th>
      <th colspan="2" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.85rem;">mantissa (2 bits)</th>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#3b82f6;color:#fff;padding:0;">S</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
      <td style="border:1px solid #1f2937;width:32px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
    </tr>
    <tr>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">7</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">6</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;"></td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">2</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">1</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">0</td>
    </tr>
  </table>
</div>

The H100 introduced FP8 tensor cores. Training with FP8 follows the same mixed-precision pattern as TF32


<div style="font-family:monospace;margin:16px 0;padding:16px;background:#1a1a2e;border-radius:8px;color:#fff;overflow-x:auto;">

<div style="font-size:12px;color:#fbbf24;margin-bottom:8px;">FP8 mixed-precision training flow</div>

<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0;font-size:11px;">
  <div style="color:#888;width:62px;">Forward:</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">master weights</div>
    <div style="color:#fff;">FP32</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">cast + scale</div>
    <div style="color:#22c55e;">FP8 E4M3</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">activations</div>
    <div style="color:#22c55e;">FP8 E4M3</div>
  </div>
  <div style="color:#888;">×</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">weights</div>
    <div style="color:#22c55e;">FP8 E4M3</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">matmul</div>
    <div style="color:#a855f7;">FP8 multiply</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">accumulate</div>
    <div style="color:#3b82f6;">FP32</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">loss</div>
    <div style="color:#fff;">FP32</div>
  </div>
</div>

<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0;font-size:11px;">
  <div style="color:#888;width:62px;">Backward:</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">gradients</div>
    <div style="color:#22c55e;">FP8 E5M2</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">dequantize</div>
    <div style="color:#3b82f6;">FP32</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">optimizer</div>
    <div style="color:#fff;">FP32</div>
  </div>
  <div style="color:#888;">→</div>
  <div style="background:#16213e;border:1px solid #0f3460;border-radius:4px;padding:6px 10px;text-align:center;">
    <div style="color:#888;font-size:9px;">update</div>
    <div style="color:#fff;">FP32 master</div>
  </div>
</div>

<div style="font-size:11px;color:#888;margin:8px 0;">
  <span style="color:#fbbf24;">scale</span> = FP8_max / tensor_amax &nbsp;•&nbsp;
  <span style="color:#fbbf24;">dequantize</span> = result / scale &nbsp;•&nbsp;
  forward uses <span style="color:#22c55e;">E4M3</span> &nbsp;•&nbsp;
  backward uses <span style="color:#22c55e;">E5M2</span>
</div>

</div>

The major new piece that allows this to work is per tensor scaling. You need to find a scale value for each tensor in order to fit it into the range for FP8

```python
# per-tensor scaling for FP8
amax = input.abs().max()              # largest value in the tensor
scale = 448.0 / amax                   # map max → FP8 max (448 for E4M3)
scaled = (input * scale).to(torch.float8_e4m3fn)   # quantize to FP8
```

The scale factor is stored alongside the FP8 tensor so the next layer knows how to interpret it. PyTorch handles this automatically with `torch.fp8` APIs.

With newer NVIDIA cards like H100 you can actually train in FP8. You don't even need to worry about the loss scaler since all tensors get scaled properly. 

### Floating Point 4 (FP4)

FP4 is experimental. It can really save you memory but in my experience some models start to show degradation at this low precision. If the model was trained using quantized aware training then it can still be pretty good. I hope we continue to research and try new things at these lower precision data types.

Here is the standard format

**E2M1** (2 exponent, 1 mantissa). With 4 bits total, you can represent exactly 16 values:

<div style="overflow-x:auto;margin:1rem 0;">
  <div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;"><strong>FP4 E2M1</strong> (1 sign | 2 exponent | 1 mantissa)</div>
  <table style="border-collapse:collapse;font-family:monospace;line-height:1.2;">
    <tr>
      <th colspan="1" style="border:1px solid #1e3a5f;padding:6px 2px;background:#2563eb;color:#fff;text-align:center;font-size:0.85rem;">sign</th>
      <th colspan="2" style="border:1px solid #14532d;padding:6px 2px;background:#16a34a;color:#fff;text-align:center;font-size:0.85rem;">exponent (2 bits)</th>
      <th colspan="1" style="border:1px solid #7f1d1d;padding:6px 2px;background:#dc2626;color:#fff;text-align:center;font-size:0.85rem;">mantissa (1 bit)</th>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;width:42px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#3b82f6;color:#fff;padding:0;">S</td>
      <td style="border:1px solid #1f2937;width:42px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:42px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#22c55e;color:#fff;padding:0;">E</td>
      <td style="border:1px solid #1f2937;width:42px;height:26px;text-align:center;font-weight:bold;font-size:0.85rem;background:#ef4444;color:#fff;padding:0;">M</td>
    </tr>
    <tr>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">3</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">2</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">1</td>
      <td style="text-align:center;font-size:0.65rem;color:#6b7280;padding:1px 0;">0</td>
    </tr>
  </table>
</div>

<div style="overflow-x:auto;margin:1rem 0;">
  <div style="font-family:monospace;font-size:0.9rem;margin-bottom:0.5rem;">All 16 representable values:</div>
  <table style="border-collapse:collapse;font-family:monospace;line-height:1.2;text-align:center;">
    <tr>
      <th style="border:1px solid #1f2937;padding:6px 10px;background:#1e3a5f;color:#fff;font-size:0.8rem;">Bits</th>
      <th style="border:1px solid #1f2937;padding:6px 12px;background:#1e3a5f;color:#fff;font-size:0.8rem;">Value</th>
      <th style="border:1px solid #1f2937;padding:6px 10px;background:#1e3a5f;color:#fff;font-size:0.8rem;">Bits</th>
      <th style="border:1px solid #1f2937;padding:6px 12px;background:#1e3a5f;color:#fff;font-size:0.8rem;">Value</th>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 00 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;background:#1a1a2e;color:#fff;">+0.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 00 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;background:#1a1a2e;color:#fff;">−0.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 00 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+0.5</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 00 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−0.5</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 01 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+1.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 01 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−1.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 01 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+1.5</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 01 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−1.5</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 10 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+2.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 10 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−2.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 10 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+3.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 10 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−3.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 11 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+4.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 11 0</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−4.0</td>
    </tr>
    <tr>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">0 11 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">+6.0</td>
      <td style="border:1px solid #1f2937;padding:4px 10px;font-size:0.8rem;">1 11 1</td>
      <td style="border:1px solid #1f2937;padding:4px 12px;font-size:0.8rem;">−6.0</td>
    </tr>
  </table>
</div>

So can you train in FP4? Not really or at least not yet. We lose so much precision that when you compute gradients you will run into too much underflow/overflow. This mode is meant to be used for inference only at the moment.

The model is trained in BF16/FP16 and then quantized and scaled to 4bit for inference.

There is some research on FP4 training that uses similar techniques to FP8. In a practical sense it is still too unstable for adoption.

Below are some papers about attempting training at FP4:

- **[FP4 All the Way](https://arxiv.org/abs/2505.19115)** (Chmiel et al., 2025)

- **[Quartet](https://arxiv.org/abs/2505.14669)** (Castro et al., 2025) 

- **[Pretraining LLMs with NVFP4](https://arxiv.org/abs/2509.25149)** (NVIDIA, 2025)

I think we will have large adoption of FP4 and other smaller precisions in the future.

## Conclusion

| Format | Best for | Hardware | Range | Notes |
|--------|----------|----------|-------|-------|
| **FP64** | Scientific computing, numerical simulation | CPUs, HPC GPUs (A100, H100, B200) | ±1.8 × 10³⁰⁸ | Overkill for ML: use only when double precision is required |
| **FP32** | Master weights, loss scaling, accumulator | All hardware | ±3.4 × 10³⁸ | The reference precision. Everything casts down from here |
| **TF32** | Training throughput on NVIDIA Tensor Cores | Ampere+ (A100, H100, B200) | ±3.4 × 10³⁸ | Transparent drop-in for FP32 in PyTorch: free speed on compatible GPUs |
| **BF16** | Training large models (7B+) | TPU v2+, Ampere+ GPUs | ±3.4 × 10³⁸ | Best range/precision trade-off for training. No loss scaling needed |
| **FP16** | Fine-tuning, inference, smaller models | Most GPUs (Pascal+) | ±65,504 | More precision than BF16 but less range: requires GradScaler for training |
| **FP8 (E4M3)** | Forward pass, inference quantization | Hopper+ (H100, B200) | ±448 | Weights and activations during FP8 training |
| **FP8 (E5M2)** | Gradients during FP8 training | Hopper+ (H100, B200) | ±57,344 | Extra range needed for backprop |
| **FP4** | Inference quantization | Blackwell B200 (experimental) | ±6 | Training still research-only. Use for memory-bound inference |

Now we have a much better understanding of how these data types work and how they can help ML training. GPUs keep improving and the hardware can be pretty complicated. 

Keep these data types in mind when you have to purchase hardware for yourself or your company. You want hardware that supports your training/inference needs. Good luck and maybe try some FP4 training on your own.