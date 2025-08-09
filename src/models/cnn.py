"""Simplified CNN model using ResNet50 for MSc coursework."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50Classifier(nn.Module):
    """ResNet50 based classifier for plant disease classification."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)
        
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the final classifier."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ResNet50 model."""
    return ResNet50Classifier(num_classes, pretrained)


def get_model_info(model: nn.Module) -> dict:
    """Get model information including parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
    }
