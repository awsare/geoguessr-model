"""
Evaluate a trained GeoLocate checkpoint on the test split.

Usage:
    python evaluate.py
"""

import os
import math
from statistics import median

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import BATCH_SIZE, CHECKPOINT_PATH, MANIFEST_PATH, TRAIN_NUM_WORKERS
from dataset import GeoLocateDataset
from model import Net
from sectors import get_active_sector_centroids
from train import get_device


def evaluate_overall(net, testloader, device):
    """Return overall accuracy (%) of net on testloader."""
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(
            testloader,
            desc="Overall Accuracy",
            unit="batch",
            leave=False,
            position=1,
        ):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise RuntimeError("Test split has zero samples; cannot compute accuracy.")
    return 100 * correct / total


def evaluate_per_class(net, testloader, label_map, device):
    """Return per-sector accuracies as sorted (sector, accuracy_pct) tuples."""
    idx_to_sector = {idx: sector for sector, idx in label_map.items()}
    correct_pred = {sector: 0 for sector in label_map}
    total_pred = {sector: 0 for sector in label_map}

    with torch.no_grad():
        for data in tqdm(
            testloader,
            desc="Per-Class Accuracy",
            unit="batch",
            leave=False,
            position=1,
        ):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                sector = idx_to_sector[label.item()]
                if label == prediction:
                    correct_pred[sector] += 1
                total_pred[sector] += 1

    results = []
    for sector, correct_count in sorted(correct_pred.items()):
        accuracy = 100 * correct_count / total_pred[sector] if total_pred[sector] else 0
        results.append((sector, accuracy))
    return results


def evaluate_confusion_matrix(net, testloader, label_map, device, output_path):
    """Save a confusion-matrix image and return its output path."""
    num_classes = len(label_map)
    idx_to_sector = {idx: sector for sector, idx in label_map.items()}
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for data in tqdm(
            testloader,
            desc="Confusion Matrix",
            unit="batch",
            leave=False,
            position=1,
        ):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for true_label, pred_label in zip(labels.cpu(), predictions.cpu()):
                cm[true_label.item(), pred_label.item()] += 1

    labels = [idx_to_sector[idx] for idx in range(num_classes)]
    fig_width = max(8, int(num_classes * 0.8))
    fig_height = max(6, int(num_classes * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm.numpy(), interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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


def evaluate_geographic_distance(net, testloader, label_map, device):
    """Return distance-aware metrics using sector-centroid great-circle error."""
    idx_to_sector = {idx: sector for sector, idx in label_map.items()}
    centroid_map = get_active_sector_centroids()
    missing = sorted(set(idx_to_sector.values()) - set(centroid_map))
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            "Missing centroid coordinates for sector(s): "
            f"{missing_names}. Update centroid maps in sectors.py."
        )

    distances_km = []
    within_500 = 0
    within_2000 = 0
    within_3000 = 0
    within_4000 = 0
    within_5000 = 0
    weighted_sum = 0.0
    weighted_tau_km = 1500.0

    with torch.no_grad():
        for data in tqdm(
            testloader,
            desc="Geographic Distance",
            unit="batch",
            leave=False,
            position=1,
        ):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for true_label, pred_label in zip(labels.cpu(), predictions.cpu()):
                true_sector = idx_to_sector[true_label.item()]
                pred_sector = idx_to_sector[pred_label.item()]
                true_lat, true_lon = centroid_map[true_sector]
                pred_lat, pred_lon = centroid_map[pred_sector]
                distance_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
                distances_km.append(distance_km)

                if distance_km <= 500:
                    within_500 += 1
                if distance_km <= 2000:
                    within_2000 += 1
                if distance_km <= 3000:
                    within_3000 += 1
                if distance_km <= 4000:
                    within_4000 += 1
                if distance_km <= 5000:
                    within_5000 += 1
                weighted_sum += math.exp(-distance_km / weighted_tau_km)

    if not distances_km:
        raise RuntimeError("No predictions available for geographic distance evaluation.")

    count = len(distances_km)
    mean_km = sum(distances_km) / count
    median_km = median(distances_km)
    weighted_score = weighted_sum / count

    return {
        "mean_km": mean_km,
        "median_km": median_km,
        "within_500_pct": 100.0 * within_500 / count,
        "within_2000_pct": 100.0 * within_2000 / count,
        "within_3000_pct": 100.0 * within_3000 / count,
        "within_4000_pct": 100.0 * within_4000 / count,
        "within_5000_pct": 100.0 * within_5000 / count,
        "weighted_score": weighted_score,
    }


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
    test_loader_kwargs = {"num_workers": TRAIN_NUM_WORKERS}
    if TRAIN_NUM_WORKERS > 0:
        test_loader_kwargs["persistent_workers"] = True
    testloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **test_loader_kwargs,
    )

    net = load_checkpoint(CHECKPOINT_PATH, len(test_dataset.label_map), device)
    print(f"Loaded model from {CHECKPOINT_PATH}")

    evaluation_steps = tqdm(total=4, desc="Evaluation Progress", unit="stage", position=0)

    overall_accuracy = evaluate_overall(net, testloader, device)
    evaluation_steps.update(1)

    per_class_results = evaluate_per_class(net, testloader, test_dataset.label_map, device)
    evaluation_steps.update(1)

    geo_metrics = evaluate_geographic_distance(net, testloader, test_dataset.label_map, device)
    evaluation_steps.update(1)

    confusion_matrix_path = os.path.join(
        os.path.dirname(CHECKPOINT_PATH),
        "confusion_matrix.png",
    )
    confusion_matrix_output = evaluate_confusion_matrix(
        net,
        testloader,
        test_dataset.label_map,
        device,
        confusion_matrix_path,
    )
    evaluation_steps.update(1)
    evaluation_steps.close()

    print("\nEvaluation Summary")
    print(f"Accuracy on test images: {overall_accuracy:.1f} %")

    print("Per-class accuracy:")
    for sector, accuracy in per_class_results:
        print(f"Accuracy for {sector:26s}: {accuracy:.1f} %")

    print("Geographic distance metrics (sector-centroid based):")
    print(f"  Mean error (km):     {geo_metrics['mean_km']:.1f}")
    print(f"  Median error (km):   {geo_metrics['median_km']:.1f}")
    print(f"  Within 500 km:       {geo_metrics['within_500_pct']:.2f}%")
    print(f"  Within 2000 km:      {geo_metrics['within_2000_pct']:.2f}%")
    print(f"  Within 3000 km:      {geo_metrics['within_3000_pct']:.2f}%")
    print(f"  Within 4000 km:      {geo_metrics['within_4000_pct']:.2f}%")
    print(f"  Within 5000 km:      {geo_metrics['within_5000_pct']:.2f}%")
    print(
        "  Distance score exp(-d/tau), tau=1500 km: "
        f"{geo_metrics['weighted_score']:.4f}"
    )
    print(f"Saved confusion matrix to {confusion_matrix_output}")


if __name__ == "__main__":
    main()