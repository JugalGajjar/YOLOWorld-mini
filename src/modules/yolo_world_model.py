"""
YOLOv8 Backbone -> RepVL-PAN -> Head
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from modules.backbone_yolov8 import YOLOv8Backbone
from modules.neck_repvl_pan import RepVLNeck
from modules.head_yoloworld import YOLOWorldHead
from modules.text_encoder import CLIPTextEncoder
from utils.config import Config


class YOLOWorldMini(nn.Module):
    """
    Small-scale YOLO-World style model:
      - YOLOv8 backbone
      - simplified RepVL-PAN neck
      - region-text contrastive head
    """

    def __init__(self, cfg: Config = Config()):
        super().__init__()
        mcfg = cfg.model

        self.backbone = YOLOv8Backbone(variant=mcfg.backbone_variant)
        channels_list = mcfg.feat_channels  # (144, 144, 144)

        self.text_encoder = CLIPTextEncoder(
            model_name=mcfg.clip_model_name,
            pretrained=mcfg.clip_pretrained,
        )

        self.neck = RepVLNeck(channels_list=channels_list, text_dim=mcfg.embed_dim)
        # For now, attaching head to P3 (highest resolution)
        self.head = YOLOWorldHead(
            in_channels=channels_list[0],
            num_anchors=mcfg.num_anchors,
            embed_dim=mcfg.embed_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
        vocab_texts: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        images: (B, 3, H, W)
        vocab_texts: list of category prompts
        device: torch.device
        """
        # 1. encode vocab
        text_embs = self.text_encoder.encode_texts(vocab_texts, device=device)  # (C_vocab, D)

        # 2. backbone features
        p3, p4, p5 = self.backbone(images)
        feats = [p3, p4, p5]

        # 3. RepVL neck
        fused_feats, updated_text = self.neck(feats, text_embs)

        # Use P3 for head
        p3_fused = fused_feats[0]

        # 4. head: boxes + logits
        boxes, logits, region_embs = self.head(p3_fused, updated_text)

        return boxes, logits, region_embs