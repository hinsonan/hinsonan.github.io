---
layout: post
title: "Batch Normalization: I just want to fit in"
date: 2025-10-03
categories: ML
---

What is going on with batch normalization in 2025? What is the purpose of this technique? Have we learned any lessons since it's inception?

# Batch Normalization

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167) started it all and introduced the popular method.

## What does this technique do?

Batch normalization shifts the mean to 0 and variance to 1 for each feature independently but that's not all. Batch normalization contains learnable parameters `gamma` and `beta`

Gamma controls the **standard deviation** of the output:

$$\gamma = 1: \text{ keeps the normalized variance of 1}$$

$$\gamma > 1: \text{ increases the spread of values}$$

$$\gamma < 1: \text{ decreases the spread of values}$$

$$\gamma = 0: \text{ collapses all values to the same point (rarely useful)}$$

Beta controls the **mean** of the output:

$$\beta = 0: \text{ keeps the normalized mean of 0}$$

$$\beta > 0: \text{ shifts the distribution to positive values}$$

$$\beta < 0: \text{ shifts the distribution to negative values}$$

These parameters can **recover the original pre-normalization distribution** if needed:

$$\gamma = \sqrt{\text{Var}[x_{\text{original}}]}$$

$$\beta = \text{E}[x_{\text{original}}]$$

This means Batch Normalization can learn to "undo" itself if the original distribution was actually optimal.

### Example

If normalized input $\hat{x} = [-1.5, -0.5, 0.5, 1.5]$ (mean=0, std=1):

With $\gamma = 2, \beta = 3$:
$$y = 2 \cdot \hat{x} + 3 = [0, 2, 4, 6] \text{ (mean=3, std=2)}$$

With $\gamma = 0.5, \beta = -1$:
$$y = 0.5 \cdot \hat{x} - 1 = [-1.75, -1.25, -0.75, -0.25] \text{ (mean=-1, std=0.5)}$$

## Why do we need these learnable params?

All that math is too hard to read after a few tylenol hit your bloodstream. Essentially if you were to make the mean 0 with a variance of 1 all the time that could result in information loss for certain layers. If larger positive values were important learned features then you will destroy that with normal standardization.

It helps keep training stable. This method means the model can gradually update `gamma` and `beta`

## Why this Technique is Important

### Smoother Training

As models update each layers inputs shift their distribution. Batch Norm allows the layers to have more consistent mean and variance that provides more stable training.

### Vanishing and Exploding Gradients

Batch Norm can allow for more stable gradients. Gradients can become very small or very large as they propagate backward which leads to unstable learning. When you use batch norm you reduce your dependence on this parameter scale issues.

### Learning Rates and Weight Initialization

Batch Norm can help reduce the need to fine-tune the learning rate. Since you can reduce your dependence on the chain of inputs this means you can get away with a higher learning rate without seeing large shifts and unstable training.

If you had poor weight initialization this technique can help with that since it normalizes the values and learns how to shift the inputs.