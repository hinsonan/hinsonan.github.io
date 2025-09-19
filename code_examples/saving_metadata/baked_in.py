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