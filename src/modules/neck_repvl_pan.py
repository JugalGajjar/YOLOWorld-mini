"""
Simplified RepVL-PAN: text-aware CSPLayer + image pooling attention
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCSPLayer(nn.Module):
    """
    Simplified Text-guided CSPLayer. Modulates visual features using a text-aware mask.
    """

    def __init__(self, channels: int, text_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.proj = None
        if channels != text_dim:
            self.proj = nn.Linear(text_dim, channels)

    def forward(self, x: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        text_embs: (C_vocab, D)
        """
        B, C, H, W = x.shape

        h = F.relu(self.conv1(x))

        if self.proj is not None:
            text_proj = self.proj(text_embs)  # (C_vocab, C)
        else:
            text_proj = text_embs  # (C_vocab, C)

        # Compute similarity of each spatial location to max text vector
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
        scores = h_flat @ text_proj.t()  # (BHW, C_vocab)
        scores, _ = scores.max(dim=-1)  # (BHW,)
        mask = torch.sigmoid(scores).view(B, 1, H, W)  # broadcast over C

        h = h * mask
        h = self.conv2(h)
        return h


class ImagePoolingAttention(nn.Module):
    """
    Simplified image-pooling attention:
    pools multi-scale features and uses them as key/value for attending over text embeddings.

    text_dim: dimension of text embeddings (e.g., 512)
    feat_dim: dimension of visual features (e.g., 144)
    """

    def __init__(self, text_dim: int, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.text_dim = text_dim
        self.feat_dim = feat_dim

        # Project pooled visual features (C_feat) -> text_dim
        self.feat_proj = nn.Linear(feat_dim, text_dim)

        # Multi-head attention works in text_dim space
        self.mha = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, text_embs: torch.Tensor, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        text_embs: (C_vocab, D)   where D = text_dim
        feats: list[(B, C_feat, H, W)]
        """
        pooled_list = []
        for x in feats:
            B, C, H, W = x.shape
            p = F.adaptive_avg_pool2d(x, (4, 4))  # (B, C_feat, 4, 4)
            p = p.view(B, C, 16).permute(0, 2, 1)  # (B, 16, C_feat)
            pooled_list.append(p)

        # Concatenate pooled features from all scales
        pooled = torch.cat(pooled_list, dim=1)  # (B, N, C_feat), N = 16 * num_scales

        # Project visual features into text embedding space
        pooled_proj = self.feat_proj(pooled)  # (B, N, text_dim)

        # Assume B == 1 for now
        kqv = pooled_proj  # (1, N, text_dim)
        W = text_embs.unsqueeze(0)  # (1, C_vocab, text_dim)

        updated, _ = self.mha(W, kqv, kqv)  # (1, C_vocab, text_dim)
        return text_embs + updated.squeeze(0)


class RepVLNeck(nn.Module):
    """
    Minimal RepVL-PAN style neck:
      - takes multi-scale features from YOLO backbone
      - applies ImagePoolingAttention once
      - applies TCSPLayer per scale
    """

    def __init__(self, channels_list, text_dim: int):
        super().__init__()
        c3, c4, c5 = channels_list

        # All three scales have the same channel count in your backbone (144)
        feat_dim = c3

        self.img_attn = ImagePoolingAttention(text_dim=text_dim, feat_dim=feat_dim)
        self.tcsps = nn.ModuleList([
            TCSPLayer(c3, text_dim),
            TCSPLayer(c4, text_dim),
            TCSPLayer(c5, text_dim),
        ])

    def forward(self, feats: List[torch.Tensor], text_embs: torch.Tensor):
        """
        feats: [P3, P4, P5], each (B, C, H, W)
        text_embs: (C_vocab, D)
        returns:
          fused_feats: list of same shape as feats
          updated_text: (C_vocab, D)
        """
        updated_text = self.img_attn(text_embs, feats)

        fused = []
        for x, tcsplayer in zip(feats, self.tcsps):
            fused.append(tcsplayer(x, updated_text))

        return fused, updated_text