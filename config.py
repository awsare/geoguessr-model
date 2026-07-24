"""Central configuration for GeoLocate.

Each constant below is intentionally documented so tuning and operational
choices live in one place.
"""

import os

# =============================
# Dataset Source And Label Space
# =============================

# Kaggle dataset identifier used by download_dataset.py.
KAGGLE_DATASET = "ubitquitin/geolocation-geoguessr-images-50k"

# Active geographic granularity for sector labels ("continent" or "subregion").
SECTOR_GRANULARITY = "subregion"


# ====================
# Paths And Artifacts
# ====================

# Root folders for generated metadata/checkpoints.
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"

# Manifest and label map file locations.
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.json")

# Main checkpoint produced by train.py and consumed by evaluate.py.
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "geolocate_net.pth")

# Optional final-epoch checkpoint for debugging/training analysis.
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "geolocate_net_last.pth")

# Smoke-test checkpoint path used for save/load roundtrip checks.
SMOKE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "smoke_test.pth")


# ==========================
# Manifest And Split Policy
# ==========================

# Minimum number of images required for a sector to be kept in the manifest.
MIN_IMAGES_PER_SECTOR = 50

# Train/val/test split ratios applied per sector.
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# Random seed used for deterministic split assignment.
SPLIT_SEED = 42

# Required schema for data/manifest.csv.
REQUIRED_COLUMNS = {"filepath", "country", "sector", "split"}


# =====================
# Image Preprocessing
# =====================

# Input image size expected by the model/transforms.
IMAGE_SIZE = 224

# ImageNet normalization constants for pretrained ResNet backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ResNet backbone variant used by model.py.
# Supported values: "resnet18", "resnet34", "resnet50".
BACKBONE_NAME = "resnet50"


# =========================
# Training Hyperparameters
# =========================

# Batch size for train/eval dataloaders.
BATCH_SIZE = 32

# Worker processes used by training/validation dataloaders.
TRAIN_NUM_WORKERS = 4

# Total training epochs across both warmup and fine-tuning phases.
NUM_EPOCHS = 40

# Number of initial epochs that train only the classifier head.
HEAD_WARMUP_EPOCHS = 5

# Learning rate for classifier-head warmup phase.
HEAD_LEARNING_RATE = 0.001

# Learning rate for backbone params during full-network fine-tuning.
BACKBONE_LEARNING_RATE = 0.00008

# Learning rate for classifier head during full-network fine-tuning.
FINETUNE_HEAD_LEARNING_RATE = 0.0004

# SGD momentum used in both training phases.
MOMENTUM = 0.9

# L2 regularization weight for optimizer parameter groups.
WEIGHT_DECAY = 1e-4

# Interval (in mini-batches) for printing running loss.
PRINT_EVERY = 100

# Enable OneCycleLR scheduling (applied separately per training phase).
USE_ONE_CYCLE_LR = True

# OneCycleLR shape controls.
# Fraction of total steps used to increase LR from initial_lr to max_lr.
ONE_CYCLE_PCT_START = 0.25
# initial_lr = max_lr / ONE_CYCLE_DIV_FACTOR
ONE_CYCLE_DIV_FACTOR = 25.0
# final_lr = initial_lr / ONE_CYCLE_FINAL_DIV_FACTOR
ONE_CYCLE_FINAL_DIV_FACTOR = 1000.0

# Validation metric used to select the best checkpoint.
# Supported values: "overall_accuracy", "macro_accuracy".
# overall_accuracy = total correct / total samples (dominated by large classes).
# macro_accuracy = average of per-class accuracies (each class weighted equally).
# Use macro_accuracy when balanced sector performance matters most.
BEST_CHECKPOINT_METRIC = "macro_accuracy"


# =================
# Class Balancing
# =================

# Whether to use inverse-frequency class weights in CrossEntropyLoss.
USE_CLASS_WEIGHTS = True

# Label smoothing applied to CrossEntropyLoss targets.
LABEL_SMOOTHING = 0.00

# Whether to oversample minority classes via WeightedRandomSampler.
USE_WEIGHTED_SAMPLER = False


# ==========================
# Distance-Aware Loss (Optional)
# ==========================

# If True, train with: total_loss = cross_entropy +
# DISTANCE_LOSS_WEIGHT * geographic_penalty.
#
# The geographic penalty uses sector-centroid distances and encourages the
# model to assign more probability mass to geographically nearby sectors.
USE_DISTANCE_LOSS = True

# Weight for the geographic penalty term in the total loss.
DISTANCE_LOSS_WEIGHT = 0.1

# Temperature (in km) for converting expected distance into a bounded penalty:
# penalty = 1 - exp(-expected_distance_km / DISTANCE_LOSS_TAU_KM)
DISTANCE_LOSS_TAU_KM = 1500.0


# =================
# Smoke Test Setup
# =================

# Smoke-test subset size per split (keeps smoke_test.py fast).
SMOKE_SAMPLE_SIZE = 32

# Batch size used by smoke_test.py.
SMOKE_BATCH_SIZE = 8
