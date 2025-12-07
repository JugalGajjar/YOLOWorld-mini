"""
Lightweight YOLO-style backbone used by YOLOWorld-mini.

NOTE: This does NOT depend on ultralytics or external YOLOv8 weights.
It is named YOLOv8Backbone for compatibility but is a standalone CNN.

Outputs:
    P3: 1/8 resolution, channels = img_feat_dim
    P4: 1/16 resolution
    P5: 1/32 resolution
"""

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOv8Backbone(nn.Module):
    def __init__(self, img_channels: int = 3, img_feat_dim: int = 256):
        super().__init__()
        base = img_feat_dim // 4  # e.g. 64 if img_feat_dim=256

        # Stem
        self.stem = ConvBNAct(img_channels, base, k=3, s=2)   # /2

        # Stages down to /32
        self.stage2 = ConvBNAct(base, base * 2, k=3, s=2)  # /4
        self.stage3 = ConvBNAct(base * 2, base * 4, k=3, s=2)  # /8  -> P3
        self.stage4 = ConvBNAct(base * 4, base * 8, k=3, s=2)  # /16 -> P4
        self.stage5 = ConvBNAct(base * 8, base * 16, k=3, s=2)  # /32 -> P5

        # Optional extra convs for capacity
        self.p3_conv = ConvBNAct(base * 4, img_feat_dim, k=3, s=1)
        self.p4_conv = ConvBNAct(base * 8, img_feat_dim, k=3, s=1)
        self.p5_conv = ConvBNAct(base * 16, img_feat_dim, k=3, s=1)

        # Expose out_channels for downstream modules (P3, P4, P5).
        self.out_channels: Tuple[int, int, int] = (img_feat_dim, img_feat_dim, img_feat_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W), e.g. 640x640

        Returns:
            [P3, P4, P5]:
                P3: (B, C, H/8,  W/8)
                P4: (B, C, H/16, W/16)
                P5: (B, C, H/32, W/32)
        """
        x = self.stem(x)  # /2
        x = self.stage2(x)  # /4
        c3 = self.stage3(x)  # /8
        c4 = self.stage4(c3)  # /16
        c5 = self.stage5(c4)  # /32

        p3 = self.p3_conv(c3)
        p4 = self.p4_conv(c4)
        p5 = self.p5_conv(c5)

        return [p3, p4, p5]