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
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    BACKBONE_LEARNING_RATE,
    BATCH_SIZE,
    BEST_CHECKPOINT_METRIC,
    CHECKPOINT_PATH,
    FINETUNE_HEAD_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    HEAD_WARMUP_EPOCHS,
    LAST_CHECKPOINT_PATH,
    MANIFEST_PATH,
    MOMENTUM,
    NUM_EPOCHS,
    ONE_CYCLE_DIV_FACTOR,
    ONE_CYCLE_FINAL_DIV_FACTOR,
    ONE_CYCLE_PCT_START,
    PRINT_EVERY,
    TRAIN_NUM_WORKERS,
    USE_ONE_CYCLE_LR,
    USE_CLASS_WEIGHTS,
    USE_WEIGHTED_SAMPLER,
    LABEL_SMOOTHING,
    WEIGHT_DECAY,
)
from dataset import GeoLocateDataset
from model import Net


def get_device():
    """Return the best available torch device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_counts(dataset):
    """Return per-class image counts aligned to label indices."""
    counts = torch.zeros(len(dataset.label_map), dtype=torch.float32)
    sector_indices = dataset.rows["sector"].map(dataset.label_map)
    for class_idx, class_count in sector_indices.value_counts().items():
        counts[int(class_idx)] = float(class_count)
    return counts


def build_class_weights(class_counts):
    """Return normalized inverse-frequency weights for CrossEntropyLoss."""
    if torch.any(class_counts <= 0):
        raise RuntimeError("Class counts must be > 0 to compute class weights.")

    weights = 1.0 / class_counts
    # Normalize to keep average gradient scale near unweighted loss.
    weights = weights / weights.mean()
    return weights


def build_weighted_sampler(dataset, class_counts):
    """Return a sampler that oversamples minority classes."""
    sample_weights = dataset.rows["sector"].map(
        lambda sector: 1.0 / float(class_counts[dataset.label_map[sector]])
    )
    sample_weights = torch.tensor(sample_weights.to_list(), dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate_accuracy(net, dataloader, device, num_classes):
    """Return (overall_accuracy, macro_accuracy) for dataloader."""
    net.eval()
    correct, total = 0, 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = net(inputs).argmax(1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for class_idx in range(num_classes):
                class_mask = labels == class_idx
                class_count = class_mask.sum().item()
                if class_count:
                    class_total[class_idx] += class_count
                    class_correct[class_idx] += (predictions[class_mask] == class_idx).sum().item()

    if total == 0:
        raise RuntimeError("Cannot evaluate accuracy on an empty dataloader.")

    overall = 100.0 * correct / total
    per_class_acc = torch.where(
        class_total > 0,
        class_correct.float() / class_total.float(),
        torch.zeros_like(class_total, dtype=torch.float32),
    )
    macro = 100.0 * per_class_acc.mean().item()
    return overall, macro


def run_training_epochs(
    net,
    trainloader,
    device,
    optimizer,
    criterion,
    num_epochs,
    epoch_offset=0,
    epoch_end_callback=None,
    scheduler=None,
):
    """Run one training phase for num_epochs."""
    for phase_epoch in range(num_epochs):
        epoch_number = epoch_offset + phase_epoch + 1
        current_lrs = ", ".join(f"{group['lr']:.6g}" for group in optimizer.param_groups)
        print(f"[epoch {epoch_number:02d}] lr={current_lrs}")

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            if i % PRINT_EVERY == PRINT_EVERY - 1:
                print(
                    f"[{epoch_number}, {i + 1:5d}] "
                    f"loss: {running_loss / PRINT_EVERY:.3f}"
                )
                running_loss = 0.0

        if epoch_end_callback is not None:
            epoch_end_callback(epoch_number)


def train(net, trainloader, valloader, device, criterion, best_checkpoint_path, num_classes):
    """Train net with head warmup then full-network fine-tuning."""
    head_epochs = min(HEAD_WARMUP_EPOCHS, NUM_EPOCHS)
    finetune_epochs = max(NUM_EPOCHS - head_epochs, 0)
    best_state = {"score": float("-inf"), "epoch": None, "overall": None, "macro": None}

    if BEST_CHECKPOINT_METRIC not in {"overall_accuracy", "macro_accuracy"}:
        raise ValueError(
            "BEST_CHECKPOINT_METRIC must be 'overall_accuracy' or 'macro_accuracy'. "
            f"Got: {BEST_CHECKPOINT_METRIC}"
        )

    def on_epoch_end(epoch_number):
        overall_acc, macro_acc = evaluate_accuracy(net, valloader, device, num_classes)
        score = overall_acc if BEST_CHECKPOINT_METRIC == "overall_accuracy" else macro_acc
        print(
            f"[val] epoch {epoch_number:02d}: overall={overall_acc:.2f}% "
            f"macro={macro_acc:.2f}% | selection={BEST_CHECKPOINT_METRIC}={score:.2f}%"
        )

        if score > best_state["score"]:
            best_state.update(
                {
                    "score": score,
                    "epoch": epoch_number,
                    "overall": overall_acc,
                    "macro": macro_acc,
                }
            )
            try:
                torch.save(net.state_dict(), best_checkpoint_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to write best checkpoint to {best_checkpoint_path}: {exc}"
                ) from exc
            print(
                f"[val] New best checkpoint at epoch {epoch_number:02d} "
                f"saved to {best_checkpoint_path}"
            )

    print(f"Phase 1/2: train classifier head for {head_epochs} epoch(s)")
    net.freeze_backbone()
    head_optimizer = optim.SGD(
        net.backbone.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    head_scheduler = None
    if USE_ONE_CYCLE_LR:
        steps_per_epoch = len(trainloader)
        if steps_per_epoch == 0:
            raise RuntimeError("Train dataloader has zero batches; cannot build OneCycleLR.")
        head_scheduler = optim.lr_scheduler.OneCycleLR(
            head_optimizer,
            max_lr=HEAD_LEARNING_RATE,
            epochs=head_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=ONE_CYCLE_PCT_START,
            div_factor=ONE_CYCLE_DIV_FACTOR,
            final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR,
        )

    run_training_epochs(
        net,
        trainloader,
        device,
        head_optimizer,
        criterion,
        head_epochs,
        epoch_end_callback=on_epoch_end,
        scheduler=head_scheduler,
    )

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
        finetune_scheduler = None
        if USE_ONE_CYCLE_LR:
            steps_per_epoch = len(trainloader)
            if steps_per_epoch == 0:
                raise RuntimeError(
                    "Train dataloader has zero batches; cannot build OneCycleLR."
                )
            finetune_scheduler = optim.lr_scheduler.OneCycleLR(
                finetune_optimizer,
                max_lr=[BACKBONE_LEARNING_RATE, FINETUNE_HEAD_LEARNING_RATE],
                epochs=finetune_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=ONE_CYCLE_PCT_START,
                div_factor=ONE_CYCLE_DIV_FACTOR,
                final_div_factor=ONE_CYCLE_FINAL_DIV_FACTOR,
            )

        run_training_epochs(
            net,
            trainloader,
            device,
            finetune_optimizer,
            criterion,
            finetune_epochs,
            epoch_offset=head_epochs,
            epoch_end_callback=on_epoch_end,
            scheduler=finetune_scheduler,
        )

    print("Finished Training")
    return best_state


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run prepare_dataset.py before training."
        )

    train_dataset = GeoLocateDataset("train")
    val_dataset = GeoLocateDataset("val")
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty. Re-run prepare_dataset.py to regenerate splits."
        )

    class_counts = compute_class_counts(train_dataset)
    class_weights = build_class_weights(class_counts)

    print(
        "Class count range (train split): "
        f"min={int(class_counts.min().item())}, max={int(class_counts.max().item())}"
    )
    print(
        "Balancing config: "
        f"USE_CLASS_WEIGHTS={USE_CLASS_WEIGHTS}, "
        f"USE_WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}, "
        f"LABEL_SMOOTHING={LABEL_SMOOTHING}"
    )
    print(
        "LR config: "
        f"USE_ONE_CYCLE_LR={USE_ONE_CYCLE_LR}, "
        f"ONE_CYCLE_PCT_START={ONE_CYCLE_PCT_START}"
    )

    sampler = build_weighted_sampler(train_dataset, class_counts) if USE_WEIGHTED_SAMPLER else None
    train_loader_kwargs = {"num_workers": TRAIN_NUM_WORKERS}
    val_loader_kwargs = {"num_workers": TRAIN_NUM_WORKERS}
    if TRAIN_NUM_WORKERS > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True
    print(
        "DataLoader config: "
        f"num_workers={TRAIN_NUM_WORKERS}, "
        f"persistent_workers={TRAIN_NUM_WORKERS > 0}"
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        **train_loader_kwargs,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **val_loader_kwargs,
    )

    num_classes = len(train_dataset.label_map)
    if num_classes < 2:
        raise RuntimeError(
            "Training requires at least 2 classes. "
            "Adjust sectoring/filtering and rebuild the manifest."
        )

    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=LABEL_SMOOTHING,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    net = Net(num_classes, pretrained=True).to(device)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_state = train(
        net,
        trainloader,
        valloader,
        device,
        criterion,
        CHECKPOINT_PATH,
        num_classes,
    )

    try:
        torch.save(net.state_dict(), LAST_CHECKPOINT_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write final checkpoint to {LAST_CHECKPOINT_PATH}: {exc}"
        ) from exc
    print(f"Final-epoch model saved to {LAST_CHECKPOINT_PATH}")

    if best_state["epoch"] is None:
        raise RuntimeError("No validation checkpoints were saved during training.")
    print(
        "Best model: "
        f"epoch={best_state['epoch']}, "
        f"overall={best_state['overall']:.2f}%, "
        f"macro={best_state['macro']:.2f}%"
    )
    print(f"Best-checkpoint model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
