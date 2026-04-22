"""
Corner detector — U-Net-lite with MobileNetV3-small encoder.

Outputs 4-channel heatmap (64×64) — one channel per corner
(top-left, top-right, bottom-right, bottom-left).

Target: ~0.4 MB INT8, ~80 ms on Snapdragon 6 Gen 1 @ 256×256 input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CornerDetector(nn.Module):
    """
    MobileNetV3-small encoder + lightweight decoder → 4-channel heatmap.

    Input:  [B, 3, 256, 256], ImageNet-normalized RGB
    Output: [B, 4, 64, 64],   per-corner heatmaps (sigmoid applied)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ---- Encoder: MobileNetV3-small ----
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        # features[0..3] → stride 8, features[0..8] → stride 16
        self.enc_s8 = nn.Sequential(*list(backbone.features.children())[:4])   # C=24
        self.enc_s16 = nn.Sequential(*list(backbone.features.children())[4:9]) # C=48

        # ---- Decoder ----
        self.up1 = _up_block(48, 32)   # s16 → s8
        self.fuse = nn.Conv2d(32 + 24, 32, kernel_size=1)
        self.up2 = _up_block(32, 16)   # s8 → s4  (64×64 when input is 256)
        self.head = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s8 = self.enc_s8(x)
        s16 = self.enc_s16(s8)

        d = self.up1(s16)
        d = torch.cat([d, s8], dim=1)
        d = self.fuse(d)
        d = self.up2(d)

        heat = self.head(d)
        return torch.sigmoid(heat)  # [B, 4, H/4, W/4]

    def count_params(self) -> float:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


def _up_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.Hardswish(inplace=True),
    )


if __name__ == "__main__":
    m = CornerDetector()
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(f"input  {tuple(x.shape)}")
    print(f"output {tuple(y.shape)}")
    print(f"params {m.count_params():.3f} M")
