"""
YOLOWorld-mini head: single-scale detection on P3.

Outputs:
    boxes_pred: (B, K, 4) in [0,1] xyxy
    cls_logits: (B, K, C_vocab) from CLIP-style similarity
    obj_logits: (B, K)
    region_embs: (B, K, text_dim)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOWorldHead(nn.Module):
    def __init__(self, in_channels: int, text_dim: int):
        super().__init__()
        self.text_dim = text_dim

        # Project P3 features to region embedding space.
        self.embed_conv = nn.Conv2d(in_channels, text_dim, kernel_size=1)

        # Box regression + objectness from region embeddings.
        self.box_head = nn.Conv2d(text_dim, 4, kernel_size=1)
        self.obj_head = nn.Conv2d(text_dim, 1, kernel_size=1)

    def forward(
        self,
        feats: List[torch.Tensor],
        text_embs: torch.Tensor,
    ):
        """
        Args:
            feats: [P3, P4, P5]; we only use P3 for now.
            text_embs: (C_vocab, text_dim), normalized.

        Returns:
            boxes_pred: (B, K, 4) in [0,1] xyxy
            cls_logits: (B, K, C_vocab)
            obj_logits: (B, K)
            region_embs: (B, K, text_dim) normalized
        """
        p3 = feats[0]  # (B, C, H, W)

        # Region embeddings
        emb = self.embed_conv(p3)  # (B, D, H, W)
        B, D, H, W = emb.shape
        K = H * W

        region_embs = emb.permute(0, 2, 3, 1).reshape(B, K, D)  # (B, K, D)
        region_embs = F.normalize(region_embs, dim=-1)

        # Bounding boxes in normalized xyxy
        box_logits = self.box_head(emb)  # (B, 4, H, W)
        box_logits = box_logits.permute(0, 2, 3, 1).reshape(B, K, 4)
        boxes_pred = box_logits.sigmoid()  # [0,1] xyxy

        # Objectness
        obj_logits = self.obj_head(emb)  # (B, 1, H, W)
        obj_logits = obj_logits.permute(0, 2, 3, 1).reshape(B, K)

        # Classification logits via CLIP-style similarity
        # text_embs: (C_vocab, D)
        text_norm = F.normalize(text_embs, dim=-1)  # safety
        cls_logits = torch.matmul(region_embs, text_norm.T)  # (B, K, C_vocab)

        return boxes_pred, cls_logits, obj_logits, region_embs