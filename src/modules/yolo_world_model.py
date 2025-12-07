"""
Top-level YOLOWorld-mini model:

    Backbone (YOLO-style CNN)
    Neck (simple channel-unifying PAN)
    Head (single-scale detection + region embeddings)
    Text encoder (CLIP)
"""

from typing import List

import torch
import torch.nn as nn

from utils.config import Config
from modules.backbone_yolov8 import YOLOv8Backbone
from modules.neck_repvl_pan import RepVLPAN
from modules.head_yoloworld import YOLOWorldHead
from modules.text_encoder import TextEncoder


class YOLOWorldMini(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        img_feat_dim = cfg.model.img_feat_dim
        text_dim = cfg.model.text_dim

        # Backbone
        self.backbone = YOLOv8Backbone(
            img_channels=3,
            img_feat_dim=img_feat_dim,
        )

        # Text encoder (frozen CLIP)
        self.text_encoder = TextEncoder(
            clip_model_name=cfg.model.clip_model_name,
            clip_pretrained_tag=cfg.model.clip_pretrained_tag,
            text_dim=text_dim,
        )

        # Neck + head
        self.neck = RepVLPAN(
            in_channels=self.backbone.out_channels,
            text_dim=text_dim,
            out_channels=img_feat_dim,
        )
        self.head = YOLOWorldHead(
            in_channels=img_feat_dim,
            text_dim=text_dim,
        )

        # Detection grid size (P3, 1/8 resolution).
        self.img_size = cfg.data.img_size
        self.H_feat = self.img_size // 8
        self.W_feat = self.img_size // 8

    def forward(
        self,
        images: torch.Tensor,
        vocab_texts: List[str],
        device: torch.device = None,
    ):
        """
        Args:
            images: (B, 3, H, W)  (H=W=cfg.data.img_size)
            vocab_texts: list of category names used as vocabulary.

        Returns:
            boxes_pred: (B, K, 4), normalized [0,1] xyxy
            cls_logits: (B, K, C_vocab)
            obj_logits: (B, K)
            region_embs: (B, K, text_dim)
            text_embs: (C_vocab, text_dim)
        """
        if device is not None:
            images = images.to(device)

        feats = self.backbone(images)  # [P3,P4,P5]

        # Encode vocabulary texts
        text_embs = self.text_encoder.encode_texts(vocab_texts)  # (C_vocab, text_dim)

        # Neck + head
        fused_feats, updated_text = self.neck(feats, text_embs)
        boxes_pred, cls_logits, obj_logits, region_embs = self.head(
            fused_feats,
            updated_text,
        )

        return boxes_pred, cls_logits, obj_logits, region_embs, updated_text