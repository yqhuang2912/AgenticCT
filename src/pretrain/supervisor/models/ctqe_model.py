import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Dict

class QEBackbone(nn.Module):
    """
    ResNet18 backbone adapted for 1-channel input.
    Falls back to a small CNN if torchvision is unavailable.
    """

    def __init__(self, pretrained: bool = True, feat_dim: int = 512):
        super().__init__()
        m = resnet18(weights="DEFAULT" if pretrained else None)

        # replace conv1 to accept 1 channel
        old = m.conv1
        m.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
        # init conv1 weights by averaging RGB weights if pretrained
        if pretrained and old.weight.shape[1] == 3:
            with torch.no_grad():
                m.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        m.fc = nn.Identity()
        self.backbone = m
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CTQEModel(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 4):
        super().__init__()
        self.encoder = QEBackbone(pretrained=pretrained)
        dim = self.encoder.out_dim

        self.head_ldct = nn.Linear(dim, num_classes)
        self.head_lact = nn.Linear(dim, num_classes)
        self.head_svct = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {
            "ldct": self.head_ldct(feat),
            "lact": self.head_lact(feat),
            "svct": self.head_svct(feat),
        }
