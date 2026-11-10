"""The classifier: an ImageNet-pretrained ResNet-50 with one output, the logit of referable retinopathy."""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class GlobalMean(nn.Module):
    """Global average pooling as a plain mean. Same output as nn.AdaptiveAvgPool2d(1), but its backward pass
    is deterministic on CUDA (the adaptive pooling backward uses atomic adds), so training is reproducible."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


class RetinaClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone.avgpool = GlobalMean()
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The 2048-dimensional pooled image representation (used by the shift tests)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def make_model(pretrained: bool = True) -> RetinaClassifier:
    return RetinaClassifier(pretrained)
