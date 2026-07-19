"""Model definitions for GeoLocate."""

import warnings

import torch.nn as nn
from torchvision import models


class Net(nn.Module):
    """ResNet-18 with a classifier head resized for active class count."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except (RuntimeError, OSError) as exc:
            # Fall back to random init if pretrained weights cannot be loaded.
            warnings.warn(
                "Could not load pretrained ResNet-18 weights; continuing with "
                f"random initialization. Details: {exc}",
                stacklevel=2,
            )
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def freeze_backbone(self):
        """Freeze feature extractor layers and keep classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze the full network for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)