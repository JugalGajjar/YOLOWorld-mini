"""
Simplified neck for YOLOWorld-mini.

Takes backbone feature maps [P3, P4, P5] and (optionally) text embeddings,
returns fused feature maps and (optionally updated) text embeddings.

Currently we just use P3 as-is and pass text embeddings through.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class RepVLPAN(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int], text_dim: int, out_channels: int):
        super().__init__()
        c3, c4, c5 = in_channels

        # Simple 1x1 convs to unify channels if needed.
        self.p3_conv = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.p4_conv = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.p5_conv = nn.Conv2d(c5, out_channels, kernel_size=1)

        self.text_dim = text_dim

    def forward(
        self,
        feats: List[torch.Tensor],
        text_embs: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            feats: [P3, P4, P5]
            text_embs: (C_vocab, text_dim)

        Returns:
            fused_feats: [P3', P4', P5'] (currently just channel-unified)
            updated_text: same as input (no cross-modal fusion yet)
        """
        p3, p4, p5 = feats
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        fused_feats = [p3, p4, p5]
        updated_text = text_embs  # Identity for now
        return fused_feats, updated_text