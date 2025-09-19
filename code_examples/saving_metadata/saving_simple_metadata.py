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