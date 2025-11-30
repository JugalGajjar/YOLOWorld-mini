"""
Region embeddings + cosine similarity to text embeddings
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOWorldHead(nn.Module):
    """
    YOLO-World style head:
      - predicts box offsets and region embeddings from a feature map
      - computes logits against a text vocabulary using cosine similarity
    """

    def __init__(self, in_channels: int, num_anchors: int = 3, embed_dim: int = 512):
        super().__init__()
        self.num_anchors = num_anchors
        self.embed_dim = embed_dim

        self.box_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.emb_conv = nn.Conv2d(in_channels, num_anchors * embed_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        feat: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feat: (B, C, H, W)
        text_embs: (C_vocab, D)

        returns:
          boxes: (B, K, 4)
          logits: (B, K, C_vocab)
          region_embs: (B, K, D)
        """
        B, _, H, W = feat.shape
        C_vocab, D = text_embs.shape

        box_raw = self.box_conv(feat)  # (B, A*4, H, W)
        emb_raw = self.emb_conv(feat)  # (B, A*D, H, W)

        box_raw = box_raw.view(B, self.num_anchors, 4, H, W)
        box_raw = box_raw.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)  # (B, K, 4)

        emb_raw = emb_raw.view(B, self.num_anchors, self.embed_dim, H, W)
        emb_raw = emb_raw.permute(0, 1, 3, 4, 2).reshape(B, -1, self.embed_dim)  # (B, K, D)

        e = F.normalize(emb_raw, dim=-1)  # (B, K, D)
        w = F.normalize(text_embs, dim=-1)  # (C_vocab, D)

        logits = self.alpha * torch.matmul(e, w.t()) + self.beta  # (B, K, C_vocab)

        return box_raw, logits, emb_raw