
# Importing necessary libraries
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models
from torchsummary import summary
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch.optim as optim
from IPython.core.interactiveshell import InteractiveShell
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(0)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Softmax probability calculation with temperature scaling
def softmax_with_temperature(logits, temperature=1.0):
    e_x = torch.exp(logits / temperature)
    return e_x / e_x.sum(dim=-1, keepdim=True)

# OOD detection based on softmax threshold
def is_ood(softmax_probs, threshold=0.8):
    max_probs, _ = torch.max(softmax_probs, dim=1)
    return max_probs < threshold

# Load FashionMNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Prepare data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Initialize and train the model
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add training loop, evaluation, etc. as per original code (not fully shown here for brevity)

# Example of using OOD detection during evaluation
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        logits = model(images)
        softmax_probs = softmax_with_temperature(logits, temperature=1.0)
        ood_flags = is_ood(softmax_probs, threshold=0.8)
        print("OOD Flags:", ood_flags)
