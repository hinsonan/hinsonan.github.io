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

For context I have used MLFLOW the most and any code snippets or examples will be using MLFLOW. All the principles written about will apply to all experiments tracking so do not feel like these are for MLFLOW only. This applies to the whole tracking process regardless of tracking tool.

# How to Save Checkpoints

A checkpoint should serve as a save point like in a video game. Users should be able to load a checkpoint and restart training where they left off. There are a few ways to do this. Saving all the items in a single `checkpoint.pt` file or a directory containing all the needed items to load.

These items are a must for saving off inside your `checkpoint.pt` or file directory

1) **Model State Dict**

2) **Optimizer State Dict**

3) **Epoch**

4) **Learning Rate**

5) **RNG states**

6) **Scheduler State Dict**

7) **Train/Val Loss**

8) **GradScaler** (Only used if training in mixed precision modes) 

## Model and Optimizer State Dict

State dictionaries are the model's and optimizer's parameters saved off into a dictionary. It maps the learned weights/biases to each tensor parameter. This structure only contains the data and does not include any of the model architecture code. In other words it's not the full pytorch model with all the logic flow and class structure.

Here is an example and notice how it is only the data and not the structure that gets saved

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize model and optimizer
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- PRINTING STATE DICTS ---

print("--- Model State Dict ---")
# The keys are the layer names, values are the tensors
for param_tensor in model.state_dict():
    print(f"{param_tensor:<20} | {str(model.state_dict()[param_tensor].size()):<20}")

print("\nRAW VIEW")
print(model.state_dict())

print("\n--- Optimizer State Dict ---")
# Optimizer state dict contains parameter groups and internal state (momentum, etc.)
for var_name in optimizer.state_dict():
    # We print just the keys or summary to avoid dumping massive tensors
    print(f"{var_name:<20} | {optimizer.state_dict()[var_name]}")

print("\nRAW VIEW")
print(optimizer.state_dict())
```

```
--- Model State Dict ---
fc1.weight           | torch.Size([5, 10]) 
fc1.bias             | torch.Size([5])     
fc2.weight           | torch.Size([1, 5])  
fc2.bias             | torch.Size([1])     

RAW VIEW
OrderedDict({'fc1.weight': tensor([[-0.2592,  0.3156, -0.2375,  0.0012, -0.0719, -0.2643, -0.1249, -0.1562,
          0.2153,  0.1542],
        [-0.1515, -0.2469,  0.2020,  0.2137, -0.2838,  0.2132, -0.3129,  0.2339,
          0.2600, -0.3131],
        [ 0.0360, -0.0867, -0.0630, -0.0877, -0.1519,  0.1055, -0.1394, -0.1187,
          0.2915,  0.2905],
        [-0.0682, -0.0612,  0.1123,  0.2090,  0.0065,  0.2447,  0.3058,  0.2431,
         -0.0303, -0.0872],
        [ 0.0643, -0.2017,  0.1179, -0.1888,  0.0942, -0.1812, -0.2684,  0.1670,
          0.0371,  0.0373]]), 'fc1.bias': tensor([ 0.1311,  0.0408,  0.0907, -0.3116, -0.1919]), 'fc2.weight': tensor([[-0.1753,  0.2590, -0.2536, -0.2127, -0.3042]]), 'fc2.bias': tensor([0.3435])})

--- Optimizer State Dict ---
state                | {}
param_groups         | [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'decoupled_weight_decay': False, 'params': [0, 1, 2, 3]}]

RAW VIEW
{'state': {}, 'param_groups': [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'decoupled_weight_decay': False, 'params': [0, 1, 2, 3]}]}
```

This is important to save off because it can be used to load the correct learned parameters back into the model. One huge mistake is using the `torch.save()` method

### HOLD THE PICKLES!

Pickles are from the devil. Python is already a fools language/tool but promoting pickles as a way to store objects is insane.

**YOU CAN"T EVER LOAD THESE THINGS ACROSS SERVICES OR LIBRARY VERSIONS.**

Pickle files save all this information like module paths, functions, etc...When you update or move folder structures in new versions pickle files will crash when trying to load them. If a function from a popular library is deprecated or moved the pickle file will break. Let's say you even remove a dependency in a feature push and try to load in an old pickle file that used it. Congrats it's broke

Avoid these devil incantations at all cost

### Loading the State Dict

When you save off the state dicts to the checkpoint you can now start training again where you left off at or you can load a different checkpoint to start training with different params.

`model.load_state_dict(state_dict)` and `optimizer.load_state_dict(state_dict)` will load the data into the model/optimizer and allow you to start training

## Scheduler State Dict

If you are using a scheduler for modifying the learning rate during training you need to save the state of it to the checkpoint. The reason for this is so that when you resume training the scheduler does not forget and resets the learning rate.

Some examples of a scheduler state dict are below

```
=== 1. StepLR State Dict ===
Current LR: [0.010000000000000002]
{'_get_lr_called_within_step': False,
 '_is_initial': False,
 '_last_lr': [0.010000000000000002],
 '_step_count': 8,
 'base_lrs': [0.1],
 'gamma': 0.1,
 'last_epoch': 7,
 'step_size': 5}

========================================

=== 2. CosineAnnealingLR State Dict ===
Current LR: [0.0505]
{'T_max': 50,
 '_get_lr_called_within_step': False,
 '_is_initial': False,
 '_last_lr': [0.0505],
 '_step_count': 26,
 'base_lrs': [0.1],
 'eta_min': 0.001,
 'last_epoch': 25}

========================================

=== 3. ReduceLROnPlateau State Dict ===
{'_last_lr': [0.05],
 'best': 0.9,
 'cooldown': 0,
 'cooldown_counter': 0,
 'default_min_lr': 0,
 'eps': 1e-08,
 'factor': 0.5,
 'last_epoch': 6,
 'min_lrs': [0],
 'mode': 'min',
 'mode_worse': inf,
 'num_bad_epochs': 1,
 'patience': 3,
 'threshold': 0.0001,
 'threshold_mode': 'rel'}
```

All of these values are important and it would be bad for your learning rate to be reset. This could cause some instability when you start training again.

## Saving Learning Rate, Epoch, Train/Val Loss

The reason for saving these is pretty simple. You want to know your current learning rate that you left off at so you can start training at that rate again or modify it. Epoch will tell you how far into training you are. If you are using step numbers instead of epoch (this is just total number of batches so `epochs * batch`) you will want to save it off along with batch size and number of epochs. This lets you recalculate the total number of steps.

Training and Validation loss are crucial to store. They let you know how the model's loss is being optimized and generally speaking the lower the loss the better. Also when you are scanning multiple checkpoints you can use the loss to help decide which checkpoint to grab. You may want to try checkpoints with different losses in order to experiment and reach the best loss in the future.

## RNG States

Saving off the Random Number Generator will help you maintain reproducibility of your experiments. If you do not do this and resume training you are breaking reality in a way and starting a whole new experiment. RNG can control the shuffling of the data, dropout nodes, random augmentations, and more. If you don'y save this off then you are corrupting your experiment.

It also helps with debugging. If you always crash at a certain batch then you need to keep the RNG the same to replicate the crash. The way the data is getting shuffled may be an issue. Without saving this off then you may not get the same behavior.

Without setting these you will not be able to reproduce experiment results

```python
import torch
import numpy as np
import random

# 1. Setup: Define a function to generate "random" values from all libraries
def get_random_values():
    return {
        "torch": torch.randn(1).item(),
        "numpy": np.random.rand(),
        "python": random.random()
    }

# Seed everything initially for demonstration
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("--- 1. Initial Run ---")
print(f"Step 1: {get_random_values()}")
print(f"Step 2: {get_random_values()}")

# --- SAVE THE STATE HERE (Simulating a checkpoint at Step 2) ---
checkpoint = {
    'torch_rng': torch.get_rng_state(),
    'numpy_rng': np.random.get_state(),
    'python_rng': random.getstate(),
    # If using GPU, you must also save: torch.cuda.get_rng_state()
    # 'cuda_rng': torch.cuda.get_rng_state() 
}

# Continue generating (The "Ground Truth" timeline)
print("--- 2. Continuing without stopping (Target behavior) ---")
print(f"Step 3: {get_random_values()}")
print(f"Step 4: {get_random_values()}")


# --- SIMULATE RESTART ---
print("\n... Crashing and Restarting ...\n")

# Reset seeds to prove we aren't just getting lucky
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Restore States
torch.set_rng_state(checkpoint['torch_rng'])
np.random.set_state(checkpoint['numpy_rng'])
random.setstate(checkpoint['python_rng'])
# if torch.cuda.is_available():
#     torch.cuda.set_rng_state(loaded_state['cuda_rng'])

print("--- 3. Resumed Run (Should match Target behavior) ---")
print(f"Step 3: {get_random_values()}")
print(f"Step 4: {get_random_values()}")
```

```
--- 1. Initial Run ---
Step 1: {'torch': 0.33669036626815796, 'numpy': 0.3745401188473625, 'python': 0.6394267984578837}
Step 2: {'torch': 0.12880940735340118, 'numpy': 0.9507143064099162, 'python': 0.025010755222666936}
--- 2. Continuing without stopping (Target behavior) ---
Step 3: {'torch': 0.23446236550807953, 'numpy': 0.7319939418114051, 'python': 0.27502931836911926}
Step 4: {'torch': 0.23033303022384644, 'numpy': 0.5986584841970366, 'python': 0.22321073814882275}

... Crashing and Restarting ...

--- 3. Resumed Run (Should match Target behavior) ---
Step 3: {'torch': 0.23446236550807953, 'numpy': 0.7319939418114051, 'python': 0.27502931836911926}
Step 4: {'torch': 0.23033303022384644, 'numpy': 0.5986584841970366, 'python': 0.22321073814882275}
```

Please do not forget about these important details when running model experiments

## Saving and Using GradScaler

GradScalers are used to help the backwards pass of a model when the model is in mixed precision modes. [Pytorch Docs](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling) go over this in good detail but essentially depending on what mode you are in (half precision float 16) some smaller gradients will underflow and be flushed to 0.

To help prevent this from happening you can scale the gradients but this scaler needs to be saved off. It is an adaptive scaler based on the gradients. Here is an example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

# 1. SETUP
# Check for GPU (GradScaler needs a GPU to actually do work, though it runs no-ops on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a simple dummy model
model = nn.Linear(10, 1).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scaler = GradScaler(enabled=torch.cuda.is_available())

# --- MAKE FAKE DATA ---
# Batch Size: 32, Features: 10
# We move it to the same device as the model
inputs = torch.randn(32, 10).to(device)
targets = torch.randn(32, 1).to(device)

print(f"Initial Scale: {scaler.get_scale()}")

# 2. TRAINING STEP
optimizer.zero_grad()

# A. Forward pass in autocast context
with torch.amp.autocast(str(device), enabled=torch.cuda.is_available()):
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)

# B. Scale the loss
# This multiplies the loss by the scale factor (initially 65536)
scaler.scale(loss).backward()

# C. Step the optimizer
scaler.step(optimizer)

# D. Update the scale factor
scaler.update()

print(f"Scale after step: {scaler.get_scale()}")
```

```
Initial Scale: 65536.0
Scale after step: 32768.0
```

Saving this scaler off is important for maintaining the gradients during training.

# Using MLFLOW to track the model checkpoints

