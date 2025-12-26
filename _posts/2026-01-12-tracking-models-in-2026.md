---
layout: post
title: "Don't Lose Your Model: Tracking ML Models in Experiments"
date: 2026-01-12
categories: ML
---

When people run experiments and train models they often do not log or save off all the important information about the model. There are many things to consider when designing tools for training models. A good training tool saves off artifacts that allow users to understand the models performance and behavior. It also allows users to fine-tune from checkpoints, optimize inferences, reproduce experiments, and more. Let's focus on the model saving. We need to do a better job tracking our models during training.

# Tools for tracking experiments

There are many tools for tracking ML experiments in 2026.

**Free and Open Source**

* [MLFLOW](https://mlflow.org/)
* [ClearML](https://github.com/clearml/clearml)
* [AIM](https://github.com/aimhubio/aim)

For context I have used MLFLOW the most and any code snippets or examples will be using MLFLOW and all the principles written about will apply to all experiments tracking so do not feel like these are for MLFLOW only. This applies to the whole tracking process regardless of tracking tool.

# How to Save Checkpoints

A checkpoint should serve as a save point like in a video game. Users should be able to load a checkpoint and restart training where they left off. There are a few ways to do this. Saving all the items in a single `checkpoint.pt` file or a directory containing all the needed items to load.

These items are a must for saving off inside your `checkpoint.pt` or file directory

1) **Model State Dict**

2) **Optimizer State Dict**

3) **Epoch**

4) **Learning Rate**

## Model State Dict




