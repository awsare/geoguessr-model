"""
Trains a CNN to classify GeoLocate images by country, following the
structure of PyTorch's "Training a Classifier" tutorial
(docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html): a
from-scratch conv net, CrossEntropyLoss + SGD, and a plain training loop
with running-loss prints.

Usage:
    python train.py

Requirements:
    pip install -r requirements.txt
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GeoLocateDataset

CHECKPOINT_PATH = os.path.join("checkpoints", "geolocate_net.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
PRINT_EVERY = 100


class Net(nn.Module):
    """Conv/pool/FC net, same shape as the tutorial's but scaled up for
    224x224 input (vs. CIFAR's 32x32) and a num_classes-way (vs. 10-way)
    output, sized to the active sector count from sectors.py.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 220 -> 110
        x = self.pool(F.relu(self.conv2(x)))  # 110 -> 106 -> 53
        x = self.pool(F.relu(self.conv3(x)))  # 53 -> 49 -> 24
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_device():
    """Return the best available torch device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(net, trainloader, device):
    """Train net for NUM_EPOCHS, printing running loss every PRINT_EVERY
    mini-batches, following the tutorial's training loop.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % PRINT_EVERY == PRINT_EVERY - 1:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}")
                running_loss = 0.0

    print("Finished Training")


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = GeoLocateDataset("train")
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(train_dataset.label_map)
    net = Net(num_classes).to(device)

    train(net, trainloader, device)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(net.state_dict(), CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
