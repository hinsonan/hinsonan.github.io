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