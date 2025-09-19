---
layout: post
title: "Sir a model has just attacked the 2nd pipeline stage: Bridging the gap to production"
date: 2025-09-19
categories: ML
---

You are about to wrap up the feature or delivery and are sitting there waiting for the pipeline to finish all the stages and pump out those sweet green pass marks so you can wrap this up and go home. It's getting towards the end of the testing stage and BAM you get hit with an error. Welcome to another hour of debugging and kicking off the pipeline again.

There are many reasons why this could happen. One less common thing talked about are the subtle changes to pre or post processing. We will cover this issue of how to keep track of your pre and post processing methods for production.

# But mah I cooked the model just like the last one

I didn't make no changes I swear. Sure I put pepperoni on the bottom and cheese on top of the pizza this time but it's still a pepperoni pizza. Yeah this don't fly chef.

The amount of scars and burn marks I have from this exact issue would be enough to call social services. Yet I will continue to run into this for the rest of my life. Every project struggles with this. So let's explain the problem

# Pre-Processing Denial

Preprocessing is simply whatever transforms or manipulations need to be applied to the data before it gets sent to the model. The issue is that these preprocessing steps need to be applied to any data sent to the model in the same order that they were performed during training.

For example for an image classifier I need to normalize the image

```python
import numpy as np
import cv2
 # Load image
image = cv2.imread(image_path)

# Step 0: swap image channels from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Resize to 256x256 first
image = cv2.resize(image, (256, 256))

# Step 2: Normalize to [0, 1]
image = image.astype(np.float32) / 255.0

# Step 3: Apply ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = (image - mean) / std

# Step 4: Convert to CHW format (channel, height, width)
image = np.transpose(image, (2, 0, 1))
```

This is a pretty common preprocessing step for image based models. The issue is you have to do these steps in this order or else bad things will happen. The model is trained on RGB images so sending in a BGR image will still "work" as in the model will run the inference. The output from this model will not be accurate and will most likely be less accurate. You need to normalize the image to be within 0-1 pixel values but that is not enough you need to apply the imagenet normalization function. The reason for this is that if you use a imagenet pretrained backbone it assumes that the image is being normalized to their standards. If each channel is not normalized in this order then the inference will be incorrect.

This is separate from the model but the model relies on these steps. The big issue is when you train the model and hand it off into the void. You hand it off to another team to integrate. This other team has no idea about the preprocessing steps. Sure you can preach about it and send them the code but somehow the preprocessing will get misapplied.

This gets even more complicated when the package that the model was trained with is different than the environment that runs in production. For instance if the production env uses PIL and not cv2 then you better not swap the channels from BGR to RGB because PIL reads images in as RGB by default.

# Post-Processing Depression

To make matters worse many times there is post processing that needs the same treatment. For object detection models you may need to reformat the bbox and image sizes. You may also need to apply non maximal suppression. Let's keep this brain dead simple and say you just need to take the `argmax` to find the index with the highest probability.

```python
# Step 0: Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=1)

# Step 1: Get predictions using argmax
predictions = torch.argmax(probabilities, dim=1)

# Step 2: Get confidence scores (max probability)
confidences = torch.max(probabilities, dim=1)[0]
```

You may laugh at how simple this is but you wont be laughing when the deadline is coming and you have to debug the model in the system. Maybe a wrong dimension got sent in or the softmax was not applied. Regardless now we have the same issue as we did in pre-processing. So how can we solve this issue of making sure that we know the exact steps for the pre and post processing of a model

# Tracking your Training: Finding Waldo

We will start at the training stage. You need to get your act together and track your experiments. When someone ask you how you made this model and you just drool and stare at them I assume that we have lost any hope of reproducing that experiment. Now when asked to train another model that is similar we can't because Jimmy didn't eat his dino nugs and just keeps talking conspiracies about a model with 100% accuracy.

There are many model tracking tools so just pick one and use it. MLFlow, Weights & Biases, etc... Shoot I don't even care if you use excel if you have literally nothing. Now tracking is not enough because many people don't log the important stuff. You still have to log items to the server.

Log your transforms, pre-processing, and post-processing sequences. It is also a good idea to log the commit hash of the repo used to train the model. Everyone talks about logging the model and metrics but they forget the important parts of how to process the data. There are many ways you can log this to the server you just need to make sure that it makes sense and you can reproduce the pre and post processing steps.

Even something as simple as a dictionary can be helpful

```json
{
    "step": 1,
    "operation": "resize",
    "parameters": {"size": resize_size, "interpolation": "bilinear"}
},
{
    "step": 2,
    "operation": "to_tensor",
    "parameters": {"scale": "[0,1]", "format": "CHW"}
},
{
    "step": 3,
    "operation": "normalize",
    "parameters": {"mean": mean, "std": std, "format": "imagenet"}
}
```

So this helps helps because now we can look at our tracking server and see how we processed the data for this model. But there are other things we should do

# Saving Metadata: Power Up your Checkpoints

Maybe just maybe in the year of 2025 you can attach important data to the model file. This crazy technology of writing data to a file is in fact available to us.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Save checkpoint with metadata
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 42,
    'preprocessing': {
        'normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'resize': (224, 224),
        'transforms': ['RandomHorizontalFlip', 'ColorJitter']
    },
    'postprocessing': {
        'output_classes': ['cat', 'dog', 'bird', 'fish'],
        'threshold': 0.5
    },
    'metadata': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'dataset': 'custom_data'
    }
}

torch.save(checkpoint, 'model.pth')

# Load checkpoint
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Preprocessing: {checkpoint['preprocessing']['normalization']}")
print(f"Output classes: {checkpoint['postprocessing']['output_classes']}")
print(f"Metadata: {checkpoint['metadata']}")
```

## TorchScript Exports

Now you save you processing steps in the meta data. The issue with this is that if you are using an exported model like TorchScript this wont work. You will need to save and load the metadata differently

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create and trace model
model = ImageClassifier()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save TorchScript model with metadata using _extra_files
metadata = {
    'epoch': 50,
    'loss': 0.089,
    'accuracy': 0.94,
    'preprocessing': {
        'input_shape': [3, 224, 224],
        'normalization': {
            'mean': [0.485, 0.456, 0.406], 
            'std': [0.229, 0.224, 0.225]
        },
        'pixel_range': [0.0, 1.0],
        'resize_method': 'bilinear',
        'center_crop': True
    },
    'postprocessing': {
        'output_classes': ['cat', 'dog', 'bird', 'fish'],
        'apply_softmax': True,
        'confidence_threshold': 0.7,
    },
    'model_info': {
        'architecture': 'ImageClassifier',
        'input_dtype': 'float32',
        'output_dtype': 'float32',
        'is_traced': True
    }
}

extra_files = {'metadata': str(metadata)}
traced_model.save('model.pt', _extra_files=extra_files)

# Load TorchScript model with metadata
extra_files = {'metadata': ''}
loaded_model = torch.jit.load('model.pt', _extra_files=extra_files)
metadata = eval(extra_files['metadata'])  # or json.loads() if using JSON format

print(f"TorchScript model loaded from epoch {metadata['epoch']}")
print(f"Input normalization: {metadata['preprocessing']['normalization']}")
print(f"Output classes: {metadata['postprocessing']['output_classes']}")
print(f"Confidence threshold: {metadata['postprocessing']['confidence_threshold']}")

# Use the loaded TorchScript model for inference
test_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = loaded_model(test_input)
    probabilities = F.softmax(output, dim=1)
    print(f"Model output shape: {output.shape}")
    print(f"Predicted probabilities: {probabilities.squeeze()}")
```

You can see that here you have to have your metadata in a json object. TorchScript requires you to save and load metadata this way. It is a little annoying having to enforce the `extra_files` keyword arg. This can be a good option to use in a deployed model that has been exported. Now you can parse the metadata and build out a pre and post processing system from it in production to enforce the operations

# Bake That Cake

Get the oven pre-heated because we are about to bake our pre/post processing into the `pt` model. This is a pretty neat way to handle this

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ModelWithProcessing(nn.Module):
    def __init__(self, base_model, preprocess_fn, postprocess_fn):
        super().__init__()
        self.base_model = base_model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        # Register normalization constants as buffers to avoid tracer warnings
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = self.preprocess_fn(x, self.mean, self.std)
        x = self.base_model(x)
        return self.postprocess_fn(x)

# Define preprocessing function
def preprocess_fn(x, mean, std):
    # Normalize to [0, 1]
    x = x.float() / 255.0
    # Apply ImageNet normalization using passed tensors
    x = (x - mean) / std
    # Resize to expected input size
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return x

# Define postprocessing function using NamedTuple for TorchScript compatibility
from collections import namedtuple

ModelOutput = namedtuple('ModelOutput', ['predictions', 'confidence', 'probabilities'])

def postprocess_fn(x):
    # Apply softmax to get probabilities
    probabilities = F.softmax(x, dim=1)
    # Get top prediction and confidence
    confidence, prediction = torch.max(probabilities, dim=1)
    return ModelOutput(predictions=prediction, confidence=confidence, probabilities=probabilities)

# Create base model and wrap it
base_model = ImageClassifier(num_classes=4)
wrapped_model = ModelWithProcessing(base_model, preprocess_fn, postprocess_fn)

# Example usage - raw image input (0-255 uint8)
raw_image = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)

# Forward pass handles everything
with torch.no_grad():
    output = wrapped_model(raw_image)
    
print(f"Prediction: {output.predictions.item()}")
print(f"Confidence: {output.confidence.item():.3f}")
print(f"All probabilities: {output.probabilities.squeeze()}")

# Can still trace the entire pipeline for TorchScript
example_input = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
traced_model = torch.jit.trace(wrapped_model, example_input)
torch.jit.save(traced_model, 'model_with_processing.pt')

print("Model with built-in processing saved!")
```

Now the model will always apply the pre and post processing. You do not have to manage the dependencies or try to recreate the pre and post processing in production. This sounds like the silver bullet. Well guess what happens if your preprocessing step can't be a torch operation. It's a hard crash. If you are using a special transform that torch cannot support then this is a no go. Any numpy, scipy, PIL, or OpenCV method calls will make this fail.

# Pre/Post Hybrid Baking

So every method has failed us so far and no silver bullet has been found. Welcome to the brutal world we live in. Why is dependency management and ML Ops so hard in 2025? It's because ML was never meant to be used in production environments. Let's be honest this field is extremely lazy with hardware and solve most of their performance issues with bloated docker images and 100k GPUs.

So with all that said we can use a hybrid approach of tracking our pre/post processing. You should bake in as much as possible and then track/log the pre/post processing that cannot be baked in.
