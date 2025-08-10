---
layout: post
title: "Traveling Loss Salesman: Visualizing Loss Landscapes "
date: 2025-08-07
categories: ML
---

Training your ML models is frustrating and sometimes it is difficult to know what is going on. Your loss is kinda trending down and your metrics still show a little improvement but you need to see more of the landscape. You might not have a solid understanding of the gradient plane and how the model is architected. Depending on the models layers you may find it easier to end up in a crappy local minima. Maybe your model is struggling to learn in general.

# Visualize the Gradient Plane

An important tool in the toolbox is opening your third eye and seeing the land that this loss scalar is traveling. You want your loss value to be as low as possible. The loss value is what you are trying to optimize and it measures the error between model predictions and truth. When you modify an existing model or create a model from scratch it can be helpful to understand the gradient land that the loss is traveling through. It's very common for your loss to get "stuck" and not make any progress. When this happens your model does not continue to learn well.

## Model too Deep, Why no Learn?

From the [original paper](https://arxiv.org/pdf/1712.09913) they show a ResNet-56 with and without skip connections

![ResNet56](/assets/images/res-net-gradient.png)

The reason why skip connections are important for ResNet is because without them it's very hard to navigate through the plane with all the peaks and valleys. Imagine that the loss value has to cross to the opposite side. It has many areas where you could get stuck. The high peaks are areas where to loss is really high and therefore the model is performing badly. There are also all these pockets of local minima that you could get stuck in and never reach the global minima.

If you have a model that looks like a crazy mountain then maybe you need to revisit you architecture. Very deep models that are fully connected quickly get wild and start looking like a journey to mordor. Another neat thing about visualizing this is you can compare how different optimizers and weight initializations affect a model.

## How Does This Work

Neural networks have many dimensions so how in the world are we going to be able to view this in a 3D space. Generally this is done by performing the following steps

1) Pick you current model parameters

2) Create a 2D grid that picks 2 random directions to travel in

3) Move along that grid space and use the loss value as the height

The paper goes into more details on how to do this but essentially you use the 2 directions of alpha and beta to travel the gradient space

**1D Case (Line):** $f(\alpha) = \mathcal{L}(\theta^* + \alpha\delta)$

**2D Case (Surface):** $f(\alpha, \beta) = \mathcal{L}(\theta^* + \alpha\delta + \beta\eta)$

Where:
- θ* ∈ ℝⁿ represents the center point (typically trained parameters)
- δ, η ∈ ℝⁿ are direction vectors in parameter space
- α, β ∈ ℝ are scalar coefficients defining the 2D grid
- 𝓛 is the loss function evaluated on the dataset

The other large component is a filter norm so the scales between models stay similar. Without this it is hard to compare models to each other since they may use a different scale or normalization of weights.

$$d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|} \cdot \|\theta_{i,j}\|$$

Where:
- d_{i,j} represents the j-th filter of the i-th layer in direction d
- θ_{i,j} represents the corresponding filter in the parameter vector
- ‖·‖ denotes the Frobenius norm

Now we can go through our model parameters and move around the gradient space to form these visuals

## Coding the Great Gradient Plane

We will use the MNIST dataset and a fully connected model to view the gradient landscape. We will create our simple model

```python
# Simple MNIST Network
class SimpleNet(nn.Module):
    def __init__(self, hidden_dim=20):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

Then we will compute the loss landscape at the initialized weights

```python
Alpha_init, Beta_init, Loss_init = compute_loss_landscape(
    model, train_loader, resolution=51, scale=1.0, use_filter_norm=True, smoothing=0.9, scale_multiplier=0.4
)

fig_init = visualize_loss_landscape(
    Alpha_init, Beta_init, Loss_init,
    title="Loss Landscape at Random Initialization"
)
fig_init.show()
```

### Gradient Plane of Initialized Weights

![inited_model](/assets/images/initialized_weights.png)

### Gradient Plane of Trained Weights

![trained_model](/assets/images/trained_weights.png)


What this shows in the simple model we are using is that the loss values did change between the initialized weights and trained weights. The wider width of this bowl suggest that small perturbations to the weights should not drastically alter performance.

Now we can do another cool thing. We can track the loss as it travels during training.

## Tracking the Loss During Training

Again we have to address the issue of these models having millions of params and high dimensions. So in order to track the loss while we train we have to use PCA to reduce the dimensions and find the most important directions. The reason we can't use random directions is because at high dimensions random directions can be perpendicular to each other. It gets too messy to try and find the path traveled. Think of it like finding waldo in a small room vs a 5 story mansion.

In order to track the trajectory we need non random directions and the PCA approach provides us a way to do this. The 2 principal components tell us the important directions.

When we track our simple model's loss we get this plot

![trajectory](/assets/images/tracking_loss.png)


What this shows is a smooth trajectory towards the deeper basin. This shows the loss continuing to travel downward is a pretty straight line. This is a good sign. Now that we have visualized the loss and the trajectory we have a better idea of how the model is learning. If a model was not learning well and we saw an odd trajectory or a chaotic loss plane then we could begin to either change optimizers, weight initializations, topology, etc...

# It All Comes Crashing Down

So if this is such a powerful visualization technique then why is it not used more in a lot of these large models we have today. The issue is once again the curse of dimensionality. If you are working on a model with several millions or billions of params then this idea of condensing it down to 2D is wild. You are missing so much of the information. You only see an incredibly small picture.

PCA is not always the best way to condense your data either. PCA is linear and for a billion parameter non linear model this has a chance to lose even more information.

The most disappointing aspect is the cool plot itself. You just spent all this time on this cool plot but it can also be your downfall. The plot can be doctored. I don't mean malicious but when you are trying to view these high dimensional spaces things can go wrong. SO lets say you are looking at your slice of the model's parameters in the center of the plot. You don't know for sure if those slices represent the whole pie well. The slice could look ok at a certain scale or seed and bad at another. The scale can really change how the plot looks and can fool you into thinking it is ok.

Another issue is you are not seeing the full landscape you are only seeing a small slice or one version of potentially infinite since models can have billions of params and many ways to slice those params into a viewable plot. Perhaps two methods or models appear the same in the plot but when scaled to full parameters they exhibit very different behaviors.

Not all layers and architecture work well with these conditions. The code I used was for a linear model and could be expanded to convolutions. Many models today are not fully connected and use multiple different stages and layers for training. This visualization would not be very suitable for these models

Depending on when and where you generate this plot it is expensive to compute and takes a while. When dealing with large models performance metrics and loss curves can be more reliable. There are other gradient techniques to take into account.

## When to View the Loss Landscape

Here are some times to use this loss visual:

1) Comparing smaller models to each other

2) When making a small custom model (not all problems need large models)

3) Studying topology and connectivity of models

## Other Solutions to Try

Here are some other solutions we will dive into later. These can be used to help you debug and trouble shoot models without running into some of the problems with loss gradient planes.

1) **Gradient norm tracking** - Monitor the magnitude of gradients during training to detect vanishing gradients (too small to learn) or exploding gradients (causing instability), helping you adjust learning rates or add normalization layers.

2) **Activation statistics** - Analyze the distribution of neuron activations across layers to identify dead neurons (always zero), saturated activations (always at maximum), or shifting distributions that indicate training instability.

3) **Gradient magnitudes per layer** - Examine how gradient strength varies across network depth to identify which layers are learning effectively and which might be stuck, particularly useful for diagnosing problems in very deep networks.

4) **Sharpness metrics** - Measure how sensitive your loss is to small parameter perturbations; flatter minima (low sharpness) typically generalize better than sharp minima, helping you choose between different trained models or optimizers.

These are some other tools that can be used to help understand the training of a model. ML is all about knowing when to use a certain tool. As always there is no free lunch.

[Code for Visuals](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/probing_gradients/mnist_gradient_loss_visuals.ipynb)