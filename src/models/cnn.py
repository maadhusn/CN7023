"""CNN model definitions using EfficientNet and MobileNet."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Small_Weights


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 based classifier."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features


class MobileNetClassifier(nn.Module):
    """MobileNet-V3-Small based classifier."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.mobilenet_v3_small(weights=weights)

        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[0].in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_model(
    model_name: str, num_classes: int = 4, pretrained: bool = True
) -> nn.Module:
    """Factory function to create models."""
    if model_name == "efficientnet_b0":
        return EfficientNetClassifier(num_classes, pretrained)
    elif model_name == "mobilenet_v3_small":
        return MobileNetClassifier(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_info(model: nn.Module) -> dict:
    """Get model information including parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
