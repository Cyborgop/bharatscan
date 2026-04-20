"""
Document-type classifier.

6 classes: {aadhaar, pan, cheque, gst_invoice, marksheet, generic}
Target: ~1.2 MB INT8, ~40 ms on Snapdragon 6 Gen 1 @ 224×224.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


DOC_CLASSES = ["aadhaar", "pan", "cheque", "gst_invoice", "marksheet", "generic"]


class DocClassifier(nn.Module):
    """MobileNetV3-small → 6-class head."""

    def __init__(self, num_classes: int = len(DOC_CLASSES), pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_feat = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_params(self) -> float:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    m = DocClassifier()
    x = torch.randn(1, 3, 224, 224)
    print(f"params {m.count_params():.3f} M")
    print(f"output {tuple(m(x).shape)}")
