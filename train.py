"""
Trains a ResNet-18 classifier for GeoLocate sector prediction using
CrossEntropyLoss + SGD and a plain training loop with running-loss
prints.

Usage:
    python train.py

Requirements:
    pip install -r requirements.txt
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GeoLocateDataset
from model import Net
from prepare_dataset import MANIFEST_PATH

CHECKPOINT_PATH = os.path.join("checkpoints", "geolocate_net.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
PRINT_EVERY = 100


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

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py before training."
        )

    train_dataset = GeoLocateDataset("train")
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty. Re-run prepare_dataset.py to regenerate splits."
        )
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(train_dataset.label_map)
    if num_classes < 2:
        raise RuntimeError(
            "Training requires at least 2 classes. "
            "Adjust sectoring/filtering and rebuild the manifest."
        )
    net = Net(num_classes).to(device)

    train(net, trainloader, device)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    try:
        torch.save(net.state_dict(), CHECKPOINT_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write checkpoint to {CHECKPOINT_PATH}: {exc}"
        ) from exc
    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
