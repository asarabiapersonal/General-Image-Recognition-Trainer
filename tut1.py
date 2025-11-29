import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# This function will be used to display an image
def imshow(img, title=None):
    """Imshow for Tensor."""
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(device," Proceed" if str(device) == "cuda:0" else " CHANGE TO GPU!!!!")


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(70),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(25088, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

net = Net()
net = net.to(device)

batch_size = 64
print(f"Batch size set: {batch_size}")

# NOTE: Make sure these paths are correct for your system
trainsetOG = ImageFolder(root="C:/Users/adams/OneDrive/Documents/AI/PetImages")
testsetOG = ImageFolder(root="C:/Users/adams/OneDrive/Documents/AI/PetImages2")

trainset = TransformedDataset(trainsetOG, transform=transform_train)
testset = TransformedDataset(testsetOG, transform=transform_test)
print("Transforms Applied")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
print("dataloaders constructed")

learning_rate = 0.001

# --- PLOTTING SETUP ---
# Lists to store plotting data
stepTracker = []
batchTracker = []
val_step = []
val_accuracy = []

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Turn on interactive mode for plotting
plt.ion()

# Create plots
fig, axis = plt.subplots(figsize=(10, 5))
ax2 = axis.twinx()

# Initialize empty plots with labels for the legend
line, = axis.plot(stepTracker, batchTracker, 'b-', label='Training Loss')
line2, = ax2.plot(val_step, val_accuracy, 'r-o', label='Validation Accuracy') # Added 'o' marker

# Add titles and labels
axis.set_title('Training Progress')
axis.set_xlabel('Training Step')
axis.set_ylabel('Loss', color='b')
ax2.set_ylabel('Accuracy (%)', color='r')

# Style ticks
axis.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
axis.grid(True)

# Create a combined legend
lines = [line, line2]
axis.legend(lines, [l.get_label() for l in lines])

class_names = trainsetOG.classes
print(str(class_names))

# --- TRAINING & VALIDATION LOOP ---
epochs = 10 # CHANGED: Increased epochs to see a line form
for epoch in range(epochs):
    net.train() # Set model to training mode
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        plot_every = 100 # Plot loss every 100 batches
        if (i + 1) % plot_every == 0:
            avg_loss = running_loss / plot_every
            current_step = i + 1 + (epoch * len(trainloader))
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')

            # Append new data points for the loss plot
            stepTracker.append(current_step)
            batchTracker.append(avg_loss)
            running_loss = 0.0

    print(f'Finished training epoch: {epoch + 1}')

    # --- VALIDATION ---
    net.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate overall accuracy for the epoch
    overall_accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {overall_accuracy:.2f} %')

    # Append data for the accuracy plot
    # We use the last step of the epoch for the x-axis value
    val_step.append(len(trainloader) * (epoch + 1))
    val_accuracy.append(overall_accuracy)

    # --- UPDATE PLOT INTERACTIVELY ---
    # Update data for both lines
    line.set_xdata(stepTracker)
    line.set_ydata(batchTracker)
    line2.set_xdata(val_step)
    line2.set_ydata(val_accuracy)

    # Rescale axes
    axis.relim()
    axis.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- FINAL DISPLAY ---
print('Finished Training')
plt.ioff() # Turn off interactive mode
plt.show() # Display the final plot