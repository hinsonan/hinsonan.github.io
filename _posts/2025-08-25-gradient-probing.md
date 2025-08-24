---
layout: post
title: "Probing Gradients 👽"
date: 2025-08-25
categories: ML
---

You wake up in a blurry haze. Zuck has unzipped his human flesh skin suit and you are face to face with the real lizard man. He begins to probe you in multiple ways. He wants all your information and he wont stop until you tell him everything. This is exactly what we are about to do to our neural networks. We are about to shine the light on them and make them confess to us why they refuse to learn.

We are going to focus on non visual methods to do this. The reason for this is that million param models are hard to visualize and in the last article we looked at some visual methods and discussed their pros and cons. These methods discussed today can be used on large models and really any gradient based learning method.

# What is Probing

Gradient Probing is a diagnostic tool used for exploring the gradients of a model based on the input. You can determine attributes such as 

* How sensitive is a model to certain inputs
* What areas of the model are learning
* Issues like vanishing and exploding gradients
* What inputs to the model matter more
* How robust in the model to perturbations

You will use this when you are debugging your models. If you can determine potential issues like exploding gradients you can know what techniques to use to fix them like clipping.

# When Should You Use This

Here are some situations where using this technique is helpful

* Loss plateaus or oscillates heavily
* Training diverges
* Loss values go to NaN
* Useful for determining if you can continue to train longer

# Vanishing and Exploding Gradients

You ever ask a woman what she wants to eat and she says she does not care but we know that's a lie. That's kinda like a vanishing gradient. The gradients gets so close to 0 that there is no learning or information gain happening. Exploding gradients are when you take her to Longhorn because she did not care and then she freaks out demanding Olive Garden. Your gradients explode to large numbers causing your weights to become unstable and preventing convergence.

It is common to use gradient norms to compare with. The reason for this is params can have multiple weights so that is a lot of values to check but the norm gives us one value to check and measures the total amount of change in a param/layer.

There are many ways to set a threshold for what qualifies as an exploding or vanishing gradient. Some model topologies are more prone to this behavior so you may need to set thresholds accordingly. You can dynamically set them during training based on current trends. You could compare other healthy training runs and use that to help guide you.

A good start is to set the vanishing threshold to something really close to 0 like `1e-10 / learning_rate` and the exploding gradient to `1.0 / learning_rate`. The reason why this works as a starting point is due to how the change of weights is done in the update step.

`weight_new = weight_old - learning_rate * gradient`

So the actual change to the weight is:

`weight_change = learning_rate * gradient`

so our thresholds are based on our learning rate which controls the rate of change.

Some example code to check this during training would be

```python
import torch
import numpy as np

def check_vanishing(model, learning_rate, threshold_factor=1e-10):
    """
    Check if gradients are vanishing.
    Returns (has_vanishing, num_affected, total_params)
    """
    threshold = threshold_factor / learning_rate
    vanishing_count = 0
    total_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            total_count += 1
            if p.grad.norm().item() < threshold:
                vanishing_count += 1
    
    has_vanishing = vanishing_count > 0
    return has_vanishing, vanishing_count, total_count


def check_exploding(model, learning_rate, threshold_factor=1.0):
    """
    Check if gradients are exploding.
    Returns (has_exploding, num_affected, total_params)
    """
    threshold = threshold_factor / learning_rate
    exploding_count = 0
    total_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            total_count += 1
            if p.grad.norm().item() > threshold:
                exploding_count += 1
    
    has_exploding = exploding_count > 0
    return has_exploding, exploding_count, total_count

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Check gradients
        has_exploding, exploding_count, total = check_exploding(model, learning_rate=0.01)
        if has_exploding:
            print("Exploding Gradients Found")
        
        has_vanishing, vanishing_count, total = check_vanishing(model, learning_rate=0.01)
        if has_vanishing:
            print("Vanishing Gradients Found")
        
        optimizer.step()
```

Here is a list of things to try for fixing these issues

## Fixing Vanishing Gradients

* Add Residual Connections
* Replace sigmoid/tanh with ReLU, LeakyReLU, etc..
* Normalization Techniques like batch, layer, group, weight normalizations
* Increase learning rate
* Gradient Accumulation
* Learning rate warmups

## Fixing Exploding Gradients

* Clip Gradients
* L2 regularization
* Dropout
* Learning rate decay


## Gradient Signal to Noise Ratio

This method measures how consistent the gradients are across the batches. If the gradients signal is stronger then that can tell you that the optimizer has a clear direction to step in. There are multiple ways to check SNR. You can take a page out of the sensor world and compute a simple snr like you would for radar signals or you can take the approach from this paper called [Understanding Why Neural Networks Generalize Well Through GSNR of Parameters](https://arxiv.org/pdf/2001.07384). From their abstract this method basically says if you can have a high GSNR then this indicates that during training you will have better generalization.

>The GSNR of a parameter is defined as the ratio between its gradient’s squared mean and
>variance, over the data distribution. Based on several approximations, we establish
>a quantitative relationship between model parameters’ GSNR and the generalization gap. This relationship indicates that larger GSNR during training process leads
>to better generalization performance

`SNR > 1` means the signal is stronger than the noise, which is good and `SNR < 1` mean the noise is taking over

Here is some pseudo code that you can use to help you compute these ratios

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def check_gradient_snr_perparam(
   model: nn.Module, 
   data_loader: DataLoader, 
   criterion: nn.Module, 
   n_batches: int = 10
) -> None:
   """Check Signal-to-Noise Ratio of gradients per parameter.
   
   Based on "Understanding Why Neural Networks Generalize Well Through GSNR"
   where GSNR(θⱼ) = E[g(θⱼ)]² / Var[g(θⱼ)]
   
   Args:
       model (nn.Module): Neural network model to analyze.
       data_loader (DataLoader): DataLoader providing batches of (input, target) pairs.
       criterion (nn.Module): Loss function used for computing gradients.
       n_batches (int): Number of batches to collect for statistics. Defaults to 10.
   
   Returns:
       None: Prints GSNR statistics for each layer.
   """
   all_grads = {}
   
   for i, (data, target) in enumerate(data_loader):
       if i >= n_batches:
           break
           
       model.zero_grad()
       loss = criterion(model(data), target)
       loss.backward()
       
       for name, param in model.named_parameters():
           if param.grad is not None:
               if name not in all_grads:
                   all_grads[name] = []
               all_grads[name].append(param.grad.clone().flatten())
   
   print("Gradient SNR (per layer aggregate):")
   for name, grads in all_grads.items():
       if len(grads) > 1:
           stacked = torch.stack(grads)  
           
           mean_per_param = stacked.mean(dim=0)
           var_per_param = stacked.var(dim=0)
           
           gsnr_per_param = (mean_per_param ** 2) / (var_per_param + 1e-8)
           
           median_gsnr = gsnr_per_param.median().item()
           mean_gsnr = gsnr_per_param.mean().item()
           
           low_gsnr = (gsnr_per_param < 1.0).float().mean().item() * 100
           
           print(f"  {name}:")
           print(f"    Median GSNR: {median_gsnr:.2f}")
           print(f"    Mean GSNR: {mean_gsnr:.2f}")
           print(f"    % params with GSNR<1: {low_gsnr:.1f}%")
           
           if median_gsnr < 1:
               print(f"    💣 Low GSNR - noisy gradients")
           else:
               print(f"    ✅ Good GSNR")


def check_gradient_snr_perlayer(
   model: nn.Module, 
   data_loader: DataLoader, 
   criterion: nn.Module, 
   n_batches: int = 10
) -> None:
   """Compute average GSNR across all parameters in a layer.
   
   Simplified version that treats all parameters in a layer together
   and reports a single GSNR value per layer.
   
   Args:
       model (nn.Module): Neural network model to analyze.
       data_loader (DataLoader): DataLoader providing batches of (input, target) pairs.
       criterion (nn.Module): Loss function used for computing gradients.
       n_batches (int): Number of batches to collect for statistics. Defaults to 10.
   
   Returns:
       None: Prints average GSNR for each layer.
   """
   all_grads = {}
   
   for i, (data, target) in enumerate(data_loader):
       if i >= n_batches:
           break
           
       model.zero_grad()
       loss = criterion(model(data), target)
       loss.backward()
       
       for name, param in model.named_parameters():
           if param.grad is not None:
               if name not in all_grads:
                   all_grads[name] = []
               all_grads[name].append(param.grad.clone())
   
   print("Gradient SNR (>1 is good):")
   for name, grads in all_grads.items():
       if len(grads) > 1:
           stacked = torch.stack(grads)
           
           stacked_flat = stacked.view(stacked.size(0), -1)
           
           mean_grad = stacked_flat.mean(dim=0)
           var_grad = stacked_flat.var(dim=0)
           gsnr = (mean_grad ** 2) / (var_grad + 1e-8)
           avg_gsnr = gsnr.mean().item()
           
           if avg_gsnr < 1:
               print(f"  💣 {name}: {avg_gsnr:.2f} (noisy)")
           else:
               print(f"  ✅ {name}: {avg_gsnr:.2f}")
```

You want a high ratio and if you have a low ratio it tells you that the noise is overwhelming and you need to think about some ways to adjust and tackle that.

* Gradient accumulation
* Lower variance optimizers
* Data augmentations or preprocessing
* Reduce learning rate

These are just a few of your options that you can try.

## Gradient Variance and Stability

Gradient variance is how much the gradients are changing across the batches during training. You are trying to calculate the Coefficient of Variation that tells you the relative variable rate. You can use this with SNR to help tell you potential issues with the training process.

Here is some pseudo code to track this

```python
def check_gradient_variance(
   model: nn.Module, 
   data_loader: DataLoader, 
   criterion: nn.Module, 
   n_batches: int = 10
) -> None:
   """Check gradient variance using coefficient of variation.
   
   Computes CV = std/mean for gradient norms across batches.
   CV < 1 indicates stable gradients, CV > 1 indicates high variance.
   
   Args:
       model (nn.Module): Neural network model to analyze.
       data_loader (DataLoader): DataLoader providing batches of (input, target) pairs.
       criterion (nn.Module): Loss function used for computing gradients.
       n_batches (int): Number of batches to collect for statistics. Defaults to 10.
   
   Returns:
       None: Prints coefficient of variation for each layer.
   """
   grad_norms = {}
   
   for i, (data, target) in enumerate(data_loader):
       if i >= n_batches:
           break
           
       model.zero_grad()
       loss = criterion(model(data), target)
       loss.backward()
       
       for name, param in model.named_parameters():
           if param.grad is not None:
               if name not in grad_norms:
                   grad_norms[name] = []
               grad_norms[name].append(param.grad.norm().item())
   
   print("Gradient Stability (CV<1 is stable):")
   for name, norms in grad_norms.items():
       if len(norms) > 1:
           mean = np.mean(norms)
           std = np.std(norms)
           cv = std / (mean + 1e-8)
           
           if cv > 1:
               print(f"  💣 {name}: CV={cv:.2f} (unstable)")
           else:
               print(f"  ✅ {name}: CV={cv:.2f}")
```

This can be another useful metric to track alongside of SNR. If you have high variance then its going to be difficult to have steady progress in your training.

## Conclusion

These are all ways in which you can explore gradients. Remember you are the investigator and it is up to you to build the tools needed to diagnose why a model may not be learning. It is difficult to know exactly why a model is struggling but you can slowly chip away at it over time. There are other techniques besides the ones mentioned in this article. Many people do not use these methods and they struggle when it comes to debugging their models. So hop aboard the next UFO and begin to probe your model.