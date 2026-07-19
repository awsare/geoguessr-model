"""
Smoke tests the full pipeline end to end on tiny data slices: dataset
loading, model forward pass, one training step, checkpoint save/load, and
evaluation. Doesn't check for good accuracy — just that nothing crashes and
shapes line up. Assumes data/manifest.csv already exists (run
prepare_dataset.py first if not).

Usage:
    python smoke_test.py
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

from dataset import GeoLocateDataset, MANIFEST_PATH
from evaluate import evaluate_overall, evaluate_per_class
from model import Net
from train import get_device, train

SAMPLE_SIZE = 32
BATCH_SIZE = 8
SMOKE_CHECKPOINT_PATH = os.path.join("checkpoints", "smoke_test.pth")


def check_manifest():
    """Fail fast with a clear message if the manifest hasn't been built."""
    assert os.path.exists(MANIFEST_PATH), (
        f"{MANIFEST_PATH} not found — run prepare_dataset.py first"
    )


def check_dataset():
    """GeoLocateDataset loads each split and yields correctly shaped batches."""
    datasets = {split: GeoLocateDataset(split) for split in ("train", "val", "test")}
    for split, ds in datasets.items():
        assert len(ds) > 0, f"{split} split is empty"

    label_map = datasets["train"].label_map
    loader = DataLoader(Subset(datasets["train"], range(SAMPLE_SIZE)), batch_size=BATCH_SIZE)
    images, labels = next(iter(loader))
    assert images.shape == (BATCH_SIZE, 3, 224, 224), f"unexpected image batch shape {images.shape}"
    assert labels.shape == (BATCH_SIZE,), f"unexpected label batch shape {labels.shape}"

    return datasets, label_map


def check_forward_pass(net, device, num_classes):
    """Net produces a (batch, num_classes) output for a dummy input."""
    dummy = torch.randn(2, 3, 224, 224).to(device)
    output = net(dummy)
    assert output.shape == (2, num_classes), f"unexpected output shape {output.shape}"


def check_train_step(net, datasets, device):
    """One training epoch over a tiny subset runs without error."""
    trainloader = DataLoader(
        Subset(datasets["train"], range(SAMPLE_SIZE)), batch_size=BATCH_SIZE, shuffle=True
    )
    import train as train_module

    train_module.NUM_EPOCHS = 1
    train_module.PRINT_EVERY = 1
    train(net, trainloader, device)


def check_checkpoint_round_trip(net, num_classes, device):
    """Saved state_dict reloads into a fresh Net with matching weights."""
    os.makedirs(os.path.dirname(SMOKE_CHECKPOINT_PATH), exist_ok=True)
    torch.save(net.state_dict(), SMOKE_CHECKPOINT_PATH)

    reloaded = Net(num_classes).to(device)
    reloaded.load_state_dict(torch.load(SMOKE_CHECKPOINT_PATH, map_location=device))

    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        assert torch.equal(net(dummy), reloaded(dummy)), "reloaded model outputs differ from original"

    os.remove(SMOKE_CHECKPOINT_PATH)


def check_evaluate(net, datasets, label_map, device):
    """Overall and per-sector evaluation run without error on a tiny subset."""
    testloader = DataLoader(Subset(datasets["test"], range(SAMPLE_SIZE)), batch_size=BATCH_SIZE)
    evaluate_overall(net, testloader, device)
    evaluate_per_class(net, testloader, label_map, device)


def main():
    device = get_device()
    print(f"Using device: {device}")

    try:
        check_manifest()
        print("[PASS] manifest exists")

        datasets, label_map = check_dataset()
        print("[PASS] dataset loading")

        num_classes = len(label_map)
        net = Net(num_classes).to(device)

        check_forward_pass(net, device, num_classes)
        print("[PASS] model forward pass")

        check_train_step(net, datasets, device)
        print("[PASS] training step")

        check_checkpoint_round_trip(net, num_classes, device)
        print("[PASS] checkpoint save/load")

        check_evaluate(net, datasets, label_map, device)
        print("[PASS] evaluation")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
