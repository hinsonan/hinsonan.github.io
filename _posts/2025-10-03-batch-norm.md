---
layout: post
title: "Batch Normalization: I just want to fit in"
date: 2025-10-03
categories: ML
---

What is going on with batch normalization in 2025? What is the purpose of this technique? Have we learned any lessons since its inception?

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

As models update, each layer's inputs shift their distribution. Batch Norm allows the layers to have more consistent mean and variance that provides more stable training.

### Vanishing and Exploding Gradients

Batch Norm can allow for more stable gradients. Gradients can become very small or very large as they propagate backward which leads to unstable learning. When you use batch norm you reduce your dependence on these parameter scale issues.

### Learning Rates and Weight Initialization

Batch Norm can help reduce the need to fine-tune the learning rate. Since you can reduce your dependence on the chain of inputs, this means you can get away with a higher learning rate without seeing large shifts and unstable training.

If you had poor weight initialization, this technique can help with that since it normalizes the values and learns how to shift the inputs.

# Does Batch Norm really Solve Internal Covariate Shift?

From the abstract of the paper

>Training Deep Neural Networks is complicated by the fact
that the distribution of each layer’s inputs changes during
training, as the parameters of the previous layers change.
This slows down the training by requiring lower learning
rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate
shift, and address the problem by normalizing layer inputs.

Batch norm was meant to solve internal covariate shift. This is where all layers have different distributions and can be harder to learn. We went over how batch norm fixes this.

We have learned a few things since 2015 and there have been challengers to this notion.

## Experiment Proving This Theory Wrong

Batch norm works very well but it's not necessarily because it solves the internal shifting.

The paper [How Does Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604) challenged the initial 2015 statement. They trained a VGG16 model in three different modes

1) standard (no batch norm)

2) batch norm

3) batch norm with noise added to each layer during training. The noise has a non-zero mean and non-unit variance

Number 3 should introduce shifting and make the model perform worse or make it harder to train. Instead it performed similar to mode two and both two and three did better than mode 1.

### My gradients still Explode

The paper [A Mean Field Theory of Batch Normalization](https://openreview.net/pdf?id=SyMDXnCcF7) notes that feed forward networks can still have exploding gradients with batch norm

Really deep batch norm models can still have exploding/vanishing gradients. This paper used plain old feed forward networks and showed that even when using batch norm you got exploding gradients no matter how you tuned the activations and other parameters.

The authors propose that this counter-intuitive finding is why skip connections work so well. Deep models that use skip connections with batch norm can be trained effectively.

## Lipschitzness

A dominant theory is that the power of batch norm is from the smoothing of the gradient.

Lipschitzness refers to how much a function can change relative to changes in its input. A function is Lipschitz continuous if there's a bound on how steeply it can change - mathematically, if `|f(x) - f(y)| ≤ L|x - y| for some constant L`.

β-smoothness is a specific type of Lipschitzness applied to gradients. A function is β-smooth if its gradients are Lipschitz continuous with constant β. In summary, this means the gradients don't change too rapidly.

With this smoother gradient flow, the model is easier to train. This theory is also supported by some newer normalization techniques that have spawned from this theory:

* **Layer Normalization**

* **Group Normalization**

* **Instance Normalization**

* **Weight Normalization**


From the plots from this [paper](https://arxiv.org/pdf/1805.11604) we can see this proven out in some experiments:

![batch norm](/assets/images/batch_norm.png)

You can see the large delta between the smoothness from batch norm and standard training.

## Three Factor Theory

I'm disappointed that at the time of writing I can't for the life of me find the full paper, but supposedly this [paper](https://www.researchgate.net/publication/374191930_Re-Thinking_the_Effectiveness_of_Batch_Normalization_and_Beyond) has a leading theory on why batch norm works so well:

1. **Gradient Lipschitz reduction** - smoother optimization landscape
2. **Reduced gradient magnitude expectation** - prevents instability  
3. **Reduced gradient variance** - more consistent updates

If someone can find this full paper, please send it to me.

This theory elaborates and explains why batch norm works in more ways than just smoothing the gradient.

# Conclusion

I think the theory is still in the air for why batch norm works so well. We know it helps create better models and provides a way for more stable training. The prevailing theory is that smoother gradients are the real gain from batch norm. This makes sense as smoother gradients are easier to test new ideas with and training is more stable.

The original proposition of fixing **internal covariate shift** does not seem to be the reason why batch norm works so well. While there are still benefits from standardization of the mean and variance, it appears the real gain is within the loss landscape, optimization geometry, and the implicit regularization effect that batch norm has.