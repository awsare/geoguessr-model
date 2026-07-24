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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

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
    DISTANCE_LOSS_TAU_KM,
    DISTANCE_LOSS_WEIGHT,
    WEIGHT_DECAY,
    USE_DISTANCE_LOSS,
)
from dataset import GeoLocateDataset
from model import Net
from sectors import get_active_sector_centroids


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


def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometers between two lat/lon points."""
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def build_sector_distance_matrix_km(label_map):
    """Return [num_classes, num_classes] centroid distance matrix in km."""
    idx_to_sector = {idx: sector for sector, idx in label_map.items()}
    centroid_map = get_active_sector_centroids()
    missing = sorted(set(idx_to_sector.values()) - set(centroid_map))
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            "Missing centroid coordinates for sector(s): "
            f"{missing_names}. Update centroid maps in sectors.py."
        )

    num_classes = len(label_map)
    distances = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for i in range(num_classes):
        sector_i = idx_to_sector[i]
        lat_i, lon_i = centroid_map[sector_i]
        for j in range(num_classes):
            sector_j = idx_to_sector[j]
            lat_j, lon_j = centroid_map[sector_j]
            distances[i, j] = haversine_km(lat_i, lon_i, lat_j, lon_j)
    return distances


class DistanceAwareCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional distance-aware regularization term."""

    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.0,
        distance_matrix_km=None,
        distance_weight=0.0,
        distance_tau_km=1500.0,
    ):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        if distance_matrix_km is not None:
            self.register_buffer("distance_matrix_km", distance_matrix_km)
        else:
            self.distance_matrix_km = None

        self.label_smoothing = label_smoothing
        self.distance_weight = float(distance_weight)
        self.distance_tau_km = float(distance_tau_km)
        if self.distance_tau_km <= 0:
            raise ValueError("DISTANCE_LOSS_TAU_KM must be > 0.")

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        if self.distance_weight <= 0 or self.distance_matrix_km is None:
            return ce_loss

        probs = torch.softmax(logits, dim=1)
        # Gather the distance row for each true label: [batch, num_classes].
        true_to_all_distances = self.distance_matrix_km[labels]
        expected_distance_km = (probs * true_to_all_distances).sum(dim=1)
        distance_penalty = (1.0 - torch.exp(-expected_distance_km / self.distance_tau_km)).mean()
        return ce_loss + self.distance_weight * distance_penalty


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
    training_bar=None,
):
    """Run one training phase for num_epochs."""
    for phase_epoch in range(num_epochs):
        epoch_number = epoch_offset + phase_epoch + 1

        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            if training_bar is not None:
                training_bar.update(1)

            if i % PRINT_EVERY == 0:
                if training_bar is not None:
                    current_lrs = ",".join(
                        f"{group['lr']:.2e}" for group in optimizer.param_groups
                    )
                    training_bar.set_postfix(
                        epoch=f"{epoch_number:02d}/{NUM_EPOCHS:02d}",
                        avg_loss=f"{running_loss / PRINT_EVERY:.3f}",
                        lr=current_lrs,
                    )
                running_loss = 0.0

        callback_metrics = None
        if epoch_end_callback is not None:
            callback_metrics = epoch_end_callback(epoch_number)

        if training_bar is not None and callback_metrics is not None:
            training_bar.set_postfix(
                epoch=f"{epoch_number:02d}/{NUM_EPOCHS:02d}",
                overall=f"{callback_metrics['overall']:.2f}%",
                macro=f"{callback_metrics['macro']:.2f}%",
                selected=f"{callback_metrics['score']:.2f}%",
            )


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

    steps_per_epoch = len(trainloader)
    if steps_per_epoch == 0:
        raise RuntimeError("Train dataloader has zero batches; cannot run training.")

    def on_epoch_end(epoch_number):
        overall_acc, macro_acc = evaluate_accuracy(net, valloader, device, num_classes)
        score = overall_acc if BEST_CHECKPOINT_METRIC == "overall_accuracy" else macro_acc
        tqdm.write(
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
            tqdm.write(
                f"[val] New best checkpoint at epoch {epoch_number:02d} "
                f"saved to {best_checkpoint_path}"
            )

        return {"overall": overall_acc, "macro": macro_acc, "score": score}

    tqdm.write(f"Phase 1/2: train classifier head for {head_epochs} epoch(s)")
    training_bar = tqdm(
        total=steps_per_epoch * NUM_EPOCHS,
        desc="Training Progress",
        unit="batch",
        position=0,
    )
    net.freeze_backbone()
    head_optimizer = optim.SGD(
        net.backbone.fc.parameters(),
        lr=HEAD_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    head_scheduler = None
    if USE_ONE_CYCLE_LR:
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
        training_bar=training_bar,
    )

    if finetune_epochs > 0:
        tqdm.write(f"Phase 2/2: fine-tune full network for {finetune_epochs} epoch(s)")
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
            training_bar=training_bar,
        )

    training_bar.close()
    tqdm.write("Finished Training")
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
        "Distance-loss config: "
        f"USE_DISTANCE_LOSS={USE_DISTANCE_LOSS}, "
        f"DISTANCE_LOSS_WEIGHT={DISTANCE_LOSS_WEIGHT}, "
        f"DISTANCE_LOSS_TAU_KM={DISTANCE_LOSS_TAU_KM}"
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

    loss_class_weights = class_weights.to(device) if USE_CLASS_WEIGHTS else None
    loss_distance_matrix = None
    if USE_DISTANCE_LOSS:
        loss_distance_matrix = build_sector_distance_matrix_km(train_dataset.label_map).to(device)

    criterion = DistanceAwareCrossEntropyLoss(
        class_weights=loss_class_weights,
        label_smoothing=LABEL_SMOOTHING,
        distance_matrix_km=loss_distance_matrix,
        distance_weight=DISTANCE_LOSS_WEIGHT if USE_DISTANCE_LOSS else 0.0,
        distance_tau_km=DISTANCE_LOSS_TAU_KM,
    )

    net = Net(num_classes, pretrained=True).to(device)
    print(f"Model backbone: {net.backbone_name}")

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
