# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # Default to 2, but allow dynamic resizing
    def __init__(self, num_classes=2): 
        super().__init__()
        # ... (Conv layers stay the same) ...
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2) 
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(65536, 512)
        self.dropout1 = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(512, 128)
        
        # DYNAMIC OUTPUT LAYER
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # ... (Forward pass stays the same) ...
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x