"""
PyTorch Dataset for GeoLocate: reads data/manifest.csv, filters to a split,
and loads/transforms images for training a country classifier.

Usage:
    python dataset.py

Requirements:
    pip install torch torchvision pillow
"""

import json
import os

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prepare_dataset import MANIFEST_PATH

LABEL_MAP_PATH = os.path.join("data", "label_map.json")
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_label_map(manifest):
    """Return a {country: index} mapping, sorted by country name.

    Loads LABEL_MAP_PATH if it already exists so indices stay stable across
    runs (checkpoints/eval depend on fixed indices); otherwise builds it from
    the full manifest and writes it out.
    """
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            return json.load(f)

    countries = sorted(manifest["country"].unique())
    label_map = {country: idx for idx, country in enumerate(countries)}

    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    return label_map


def build_transforms(split):
    """Return the torchvision transform pipeline for the given split.

    train gets augmentation (random crop/flip/color jitter); val/test get a
    deterministic resize + center crop. Both normalize to ImageNet
    statistics since the eventual model is a pretrained backbone.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


class GeoLocateDataset(Dataset):
    """Images + country labels for one split (train/val/test) of the manifest."""

    def __init__(self, split, manifest_path=MANIFEST_PATH, transform=None):
        manifest = pd.read_csv(manifest_path)
        # Build the label map from the full manifest, not the filtered split,
        # so train/val/test datasets always agree on indices even if a rare
        # country is missing from one split.
        self.label_map = build_label_map(manifest)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform if transform is not None else build_transforms(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = self.label_map[row["country"]]
        return image, label


def main():
    datasets = {split: GeoLocateDataset(split) for split in ("train", "val", "test")}

    num_classes = len(datasets["train"].label_map)
    print(f"Classes: {num_classes}")
    for split, ds in datasets.items():
        print(f"{split}: {len(ds)} images")

    loader = DataLoader(datasets["train"], batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    print(f"Batch image shape: {tuple(images.shape)}")
    print(f"Batch label shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
