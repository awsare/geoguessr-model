"""Model definitions for GeoLocate."""

import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    """ResNet-18 with a classifier head resized for active class count."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)