"""
Evaluate a trained GeoLocate checkpoint on the test split.

Usage:
    python evaluate.py
"""

import os

import torch
from torch.utils.data import DataLoader

from dataset import GeoLocateDataset, MANIFEST_PATH
from model import Net
from train import BATCH_SIZE, CHECKPOINT_PATH, get_device


def evaluate_overall(net, testloader, device):
    """Print overall accuracy of net on testloader."""
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise RuntimeError("Test split has zero samples; cannot compute accuracy.")
    print(f"Accuracy on test images: {100 * correct / total:.1f} %")


def evaluate_per_class(net, testloader, label_map, device):
    """Print per-sector accuracy of net on testloader."""
    idx_to_sector = {idx: sector for sector, idx in label_map.items()}
    correct_pred = {sector: 0 for sector in label_map}
    total_pred = {sector: 0 for sector in label_map}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                sector = idx_to_sector[label.item()]
                if label == prediction:
                    correct_pred[sector] += 1
                total_pred[sector] += 1

    for sector, correct_count in sorted(correct_pred.items()):
        accuracy = 100 * correct_count / total_pred[sector] if total_pred[sector] else 0
        print(f"Accuracy for {sector:26s}: {accuracy:.1f} %")


def load_checkpoint(checkpoint_path, num_classes, device):
    """Load a saved checkpoint into a fresh Net instance."""
    net = Net(num_classes).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Failed to load checkpoint. It may be corrupt or incompatible "
            "with the current model/dataset configuration. "
            f"Checkpoint: {checkpoint_path}. Details: {exc}"
        ) from exc
    net.eval()
    return net


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run train.py first to create it."
        )
        return
    if not os.path.exists(MANIFEST_PATH):
        print(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py "
            "before evaluation."
        )
        return

    test_dataset = GeoLocateDataset("test")
    if len(test_dataset) == 0:
        print("Test split is empty. Re-run prepare_dataset.py to regenerate splits.")
        return
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    net = load_checkpoint(CHECKPOINT_PATH, len(test_dataset.label_map), device)
    print(f"Loaded model from {CHECKPOINT_PATH}")

    evaluate_overall(net, testloader, device)
    evaluate_per_class(net, testloader, test_dataset.label_map, device)


if __name__ == "__main__":
    main()