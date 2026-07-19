"""
Trains a ResNet-18 classifier for GeoLocate sector prediction using
a two-phase fine-tuning schedule:
1) train classifier head with frozen backbone,
2) unfreeze full network and fine-tune end-to-end.

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
HEAD_WARMUP_EPOCHS = 3
HEAD_LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.0001
FINETUNE_HEAD_LEARNING_RATE = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
PRINT_EVERY = 100


def get_device():
    """Return the best available torch device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_training_epochs(net, trainloader, device, optimizer, criterion, num_epochs, epoch_offset=0):
    """Run one training phase for num_epochs."""
    for phase_epoch in range(num_epochs):
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
                print(
                    f"[{epoch_offset + phase_epoch + 1}, {i + 1:5d}] "
                    f"loss: {running_loss / PRINT_EVERY:.3f}"
                )
                running_loss = 0.0


def train(net, trainloader, device):
    """Train net with head warmup then full-network fine-tuning."""
    criterion = nn.CrossEntropyLoss()

    head_epochs = min(HEAD_WARMUP_EPOCHS, NUM_EPOCHS)
    finetune_epochs = max(NUM_EPOCHS - head_epochs, 0)

    print(f"Phase 1/2: train classifier head for {head_epochs} epoch(s)")
    net.freeze_backbone()
    head_optimizer = optim.SGD(
        net.backbone.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    run_training_epochs(net, trainloader, device, head_optimizer, criterion, head_epochs)

    if finetune_epochs > 0:
        print(f"Phase 2/2: fine-tune full network for {finetune_epochs} epoch(s)")
        net.unfreeze_backbone()
        backbone_params = [
            param
            for name, param in net.backbone.named_parameters()
            if not name.startswith("fc.")
        ]
        finetune_optimizer = optim.SGD(
            [
                {"params": backbone_params, "lr": BACKBONE_LEARNING_RATE},
                {"params": net.backbone.fc.parameters(), "lr": FINETUNE_HEAD_LEARNING_RATE},
            ],
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        run_training_epochs(
            net,
            trainloader,
            device,
            finetune_optimizer,
            criterion,
            finetune_epochs,
            epoch_offset=head_epochs,
        )

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
    net = Net(num_classes, pretrained=True).to(device)

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
