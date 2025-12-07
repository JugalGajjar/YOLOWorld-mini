"""
Text encoder wrapper around open_clip for YOLOWorld-mini.
"""

from typing import List

import torch
import torch.nn as nn
import open_clip


class TextEncoder(nn.Module):
    def __init__(self, clip_model_name: str, clip_pretrained_tag: str, text_dim: int):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained_tag
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Freeze CLIP weights; we only learn detection head.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # CLIP text projection dim
        with torch.no_grad():
            # text_projection is (d_model, text_dim)
            proj = self.model.text_projection
            clip_dim = proj.shape[1] if proj.ndim == 2 else proj.shape[0]

        if clip_dim != text_dim:
            self.proj = nn.Linear(clip_dim, text_dim)
        else:
            self.proj = nn.Identity()

        self.text_dim = text_dim

    @torch.no_grad()
    def encode_raw(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts with frozen CLIP model (returns normalized embeddings).
        """
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts and project to text_dim, normalized.

        Args:
            texts: list of strings, length C_vocab

        Returns:
            (C_vocab, text_dim) tensor
        """
        with torch.no_grad():
            raw = self.encode_raw(texts)  # (C_vocab, clip_dim)
        out = self.proj(raw)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out