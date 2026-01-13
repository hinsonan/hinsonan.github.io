---
layout: post
title: "Tracking ML Models: Designing Good Checkpoints"
date: 2026-01-10
categories: ML
---

I've got sweat pouring down my face. My arms are heavy, knees weak, and the model ain't ready. This is how it feels when tasked with recreating or modifying an experiment. I go to load the model and the doom sets in as I notice none of the valuable information was stored in the checkpoint file and I cannot find it anywhere else in all of the stored artifacts for the experiment.

When people run experiments and train models they often do not log or save all the important information about the model. There are many things to consider when designing tools for training models. A good training tool saves artifacts that allow users to understand the models performance and behavior. It also allows users to fine-tune from checkpoints, optimize inferences, reproduce experiments, and more. Let's focus on the model checkpoint in this article and how to effectively track the model as it trains.

# Tools for tracking experiments

There are many tools for tracking ML experiments in 2026.

**Free and Open Source**

* [MLFLOW](https://mlflow.org/)
* [ClearML](https://github.com/clearml/clearml)
* [AIM](https://github.com/aimhubio/aim)

For context I have used MLFLOW the most and any code snippets or examples will be using MLFLOW. All the principles written about will apply to all experiments tracking so do not feel like these are for MLFLOW only. This applies to the whole tracking process regardless of tracking tool.

# How to Save Checkpoints

A checkpoint should serve as a save point like in a video game. Users should be able to load a checkpoint and restart training where they left off. There are a few ways to do this. Saving all the items in a single `checkpoint.pt` file or a directory containing all the needed items to load.

These items are a must for saving inside your `checkpoint.pt` or file directory

1) **Model State Dict**

2) **Optimizer State Dict**

3) **Epoch**

4) **Learning Rate**

5) **RNG states**

6) **Scheduler State Dict**

7) **Train/Val Loss**

8) **GradScaler** (Only used if training in mixed precision modes) 

## Model and Optimizer State Dict

State dictionaries are the model's and optimizer's parameters saved into a dictionary. It maps the learned weights/biases to each tensor parameter. This structure only contains the data and does not include any of the model architecture code. In other words it's not the full PyTorch model with all the logic flow and class structure.

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

This is important to save because it can be used to load the correct learned parameters back into the model. One huge mistake is using the `torch.save()` method

### HOLD THE PICKLES!

Pickles are from the devil. Python is already a fool's language/tool but promoting pickles as a way to store objects is insane.

**YOU CAN'T EVER LOAD THESE THINGS ACROSS SERVICES OR LIBRARY VERSIONS.**

Pickle files save all this information like module paths, functions, etc...When you update or move folder structures in new versions pickle files will crash when trying to load them. If a function from a popular library is deprecated or moved the pickle file will break. Let's say you even remove a dependency in a feature push and try to load in an old pickle file that used it. Congrats it's broke

Avoid these devil incantations at all cost

### Loading the State Dict

When you save the state dicts to the checkpoint you can now start training again where you left off or you can load a different checkpoint to start training with different params.

`model.load_state_dict(state_dict)` and `optimizer.load_state_dict(state_dict)` will load the data into the model/optimizer and allow you to start training

## Scheduler State Dict

If you are using a scheduler for modifying the learning rate during training you need to save its state to the checkpoint. The reason for this is so that when you resume training the scheduler does not forget and reset the learning rate.

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

The reason for saving these is pretty simple. You want to know your current learning rate that you left off at so you can start training at that rate again or modify it. Epoch will tell you how far into training you are. If you are using step numbers instead of epoch (this is just total number of batches so `epochs * batch`) you will want to save it along with batch size and number of epochs. This lets you recalculate the total number of steps.

Training and Validation loss are crucial to store. They let you know how the model's loss is being optimized and generally speaking the lower the loss the better. Also when you are scanning multiple checkpoints you can use the loss to help decide which checkpoint to grab. You may want to try checkpoints with different losses in order to experiment and reach the best loss in the future.

## RNG States

Saving the Random Number Generator will help you maintain reproducibility of your experiments. If you do not do this and resume training you are breaking reality in a way and starting a whole new experiment. RNG can control the shuffling of the data, dropout nodes, random augmentations, and more. If you don't save this then you are corrupting your experiment.

It also helps with debugging. If you always crash at a certain batch then you need to keep the RNG the same to replicate the crash. The way the data is getting shuffled may be an issue. Without saving this you may not get the same behavior.

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

GradScalers are used to help the backwards pass of a model when the model is in mixed precision modes. [PyTorch Docs](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling) go over this in good detail but essentially depending on what mode you are in (half precision float 16) some smaller gradients will underflow and be flushed to 0.

To help prevent this from happening you can scale the gradients but this scaler needs to be saved. It is an adaptive scaler based on the gradients. Here is an example

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

Saving this scaler is important for maintaining the gradients during training.

# Using MLFLOW to track the model checkpoints

Let's make a dummy experiment that will track and store our model to a local directory with MLFLOW.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import mlflow
import os
import pprint

RUN_ID = None
# --- 1. CONFIGURATION ---
config = {
    "experiment_name": "Gradient_Rage",
    "run_name": "Saving_Model_Checkpoint",
    "input_dim": 10,
    "hidden_dim": 32,
    "output_dim": 1,
    "lr": 0.01,
    "epochs": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 2. MODEL DEFINITION ---
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. TRAINING FUNCTION ---
def train():
    global RUN_ID
    # A. Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config["experiment_name"])

    # B. Start the Run -> Capture 'run' object
    with mlflow.start_run(run_name=config["run_name"]) as run:
        
        # --- CAPTURE RUN ID ---
        RUN_ID = run.info.run_id
        print(f"Active Run ID: {RUN_ID}")
        
        # 1. Log the Model Config
        mlflow.log_params(config)
        
        # 2. Initialize Components
        model = SimpleModel(config['input_dim'], config['hidden_dim'], config['output_dim']).to(config['device'])
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        scaler = GradScaler(enabled=(config['device'] == 'cuda'))

        print(f"Starting Training on {config['device']}...")
        
        # 3. Training Loop
        for epoch in range(config['epochs']):
            model.train()
            
            inputs = torch.randn(32, config['input_dim']).to(config['device'])
            targets = torch.randn(32, config['output_dim']).to(config['device'])

            optimizer.zero_grad()

            with torch.amp.autocast(config['device'], enabled=(config['device'] == 'cuda')):
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 4. Log Metrics
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # --- 4. THE SAVE ---
        print("\nSaving Checkpoint...")
        
        checkpoint = {
            'run_id': RUN_ID,  # <--- SAVE ID HERE for traceability
            'epoch': config['epochs'],
            'config': config, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'rng_state': torch.get_rng_state(),
        }
        pprint.pprint(checkpoint)

        local_path = "checkpoint.pth"
        torch.save(checkpoint, local_path)
        mlflow.log_artifact(local_path)
        os.remove(local_path)
        
        print(f"Run {RUN_ID} Complete. Checkpoint saved to MLflow.")

train()
```

After running this you should see the run id that everything got stored to `Run 5b23bd5c2209458dbc812114230b92d4 Complete. Checkpoint saved to MLflow.` If you have mlflow installed you can check the UI to see the run.

`mlflow ui --backend-store-uri file:///<path_to_absolute_file_path_named_mlruns>`

This will open a webserver on `localhost:5000`

Now lets create a function to load this stored checkpoint

```python
def resume_from_mlflow(run_id):
    print(f"\n--- [STEP B] Resuming from MLflow ID: {run_id} ---")
    
    # 1. Download
    mlflow.set_tracking_uri("file:./mlruns")
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="checkpoint.pth")
    
    # 2. Load File
    checkpoint = torch.load(local_path)
    config = checkpoint['config']
    
    # 3. Re-Init Architecture
    model = SimpleModel(config['input_dim'], config['hidden_dim'], config['output_dim'])
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = GradScaler(enabled=(config['device'] == "cuda"))
    
    # 4. Load States
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    
    return model, optimizer, scheduler, scaler, config

model, opt, sched, scaler, config = resume_from_mlflow(run_id=RUN_ID)
    
print("\n--- [INSPECTION] Verifying Loaded Components ---")

# 1. CONFIG
print(f"1. Config Loaded: {config}")

# 2. MODEL WEIGHTS (Check first layer weights)
print(f"2. Model Weights (First 3 of Layer 1): {model.net[0].weight.view(-1)[:3].tolist()}")

# 3. OPTIMIZER (Check Param Groups)
print(f"3. Optimizer LR (should be decayed): {opt.param_groups[0]['lr']}")

# 4. SCHEDULER (Check Epoch Counter)
print(f"4. Scheduler Last Epoch: {sched.last_epoch}")

# 5. SCALER (Check Scale Factor)
print(f"5. Scaler Scale Factor: {scaler.get_scale()}")

# 6. RNG (Check a random number generation)
print(f"6. RNG Test (Next Random Num): {torch.randn(1).item()}")
```

```
--- [INSPECTION] Verifying Loaded Components ---
1. Config Loaded: {'experiment_name': 'Gradient_Rage_Demo', 'run_name': 'Run_001_Robust_Save', 'input_dim': 10, 'hidden_dim': 32, 'output_dim': 1, 'lr': 0.01, 'epochs': 5, 'device': 'cuda'}
2. Model Weights (First 3 of Layer 1): [-0.2047610729932785, -0.1341530978679657, 0.09505394101142883]
3. Optimizer LR (should be decayed): 0.0001
4. Scheduler Last Epoch: 5
5. Scaler Scale Factor: 65536.0
6. RNG Test (Next Random Num): 1.3229471445083618
```

Now we can see that the model and all the parameters are based on the checkpoint values. This shows you how to load the checkpoint and now you are set up for continued training.

# Conclusion

There is so much that goes into a good ML experiment. It takes a lot of effort to make solid training tools for teams to use. This is just one checkpoint file that we talked about. This is an important piece to get right and there are so many other items related to the model that we will talk about at a later time. Remember pickles are from the devil and always save good checkpoints.

[Code used in Article](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/tracking-models-designing-checkpoints/tracking_models.ipynb)
