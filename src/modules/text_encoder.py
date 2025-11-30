"""
CLIP text encoder wrapper using open_clip_torch
"""

from typing import List

import torch
import torch.nn as nn
import open_clip


class CLIPTextEncoder(nn.Module):
    """
    Wrapper around an OpenCLIP text encoder.
    Provides encode_texts() -> normalized embeddings.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.output_dim = model.text_projection.shape[1]

        # Freeze by default
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        returns: (C_vocab, D) normalized embeddings
        """
        tokens = self.tokenizer(texts).to(device)
        text_features = self.model.encode_text(tokens)  # (C_vocab, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features