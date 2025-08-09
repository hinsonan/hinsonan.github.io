---
layout: post
title: "Traveling Loss Salesman: Visualizing Loss Landscapes "
date: 2025-08-07
categories: ML
---

Training your ML models is frustrating and sometimes it is difficult to know what is going on. Your loss is kinda trending down and your metrics still show a little improvement but you need to see more of the landscape. You might not have a solid understanding of the gradient plane and how the model is architected. Depending on the models layers you may find it easier to end up in a crappy local minima. Maybe your model is struggling to learn in general.

# Visual the Gradient Plane

An important tool in the toolbox is opening your third eye and seeing the land that this loss scalar is traveling. You want your loss value to be as low as possible. The loss value is what you are trying to optimize and it measures the error between model predictions and truth. When you modify an existing model or create a model from scratch it can be helpful to understand the gradient land that the loss is traveling through. It's very common for your loss to get "stuck" and not make any progress. When this happens your model does not continue to learn well.

## Model too Deep, Why no Learn?

From the [original paper](https://arxiv.org/pdf/1712.09913) they show a ResNet-56 with and without skip connections

![ResNet56](/assets/images/res-net-gradient.png)

The reason why skip connections are important for ResNet is because without them it's very hard to navigate through the plane with all the peaks and valleys. Imagine that the loss value has to cross to the opposite side. It has many areas where you could get stuck. The high peaks are areas where to loss is really high and therefore the model is performing badly. There are also all these pockets of local minimas that you could get stuck in and never reach the global minima.

If you have a model that looks like a crazy mountain then maybe you need to revisit you architecture

