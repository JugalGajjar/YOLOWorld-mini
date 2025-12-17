"""
Text Encoder using CLIP for YOLO-World
Encodes category names into text embeddings
"""

import torch
import torch.nn as nn
import clip
from typing import List, Optional


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder for encoding category names
    """
    def __init__(self, model_name: str = "ViT-B/32", freeze: bool = True, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        self.clip_model, _ = clip.load(model_name, device=device)
        self.text_encoder = self.clip_model
        
        # Get text embedding dimension
        if "ViT-B" in model_name:
            self.text_dim = 512
        elif "ViT-L" in model_name:
            self.text_dim = 768
        else:
            self.text_dim = 512
        
        # Freeze CLIP if specified
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        print(f"Loaded CLIP model: {model_name} (text_dim={self.text_dim})")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text strings into embeddings
        
        Args:
            texts: List of text strings
        
        Returns:
            text_embeddings: Tensor of shape (N, text_dim)
        """
        # Tokenize texts
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Encode with CLIP
        with torch.no_grad() if not self.training else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.float()
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Forward pass"""
        return self.encode_text(texts)
    
    @property
    def embedding_dim(self) -> int:
        """Return text embedding dimension"""
        return self.text_dim


class TextAdapter(nn.Module):
    """
    Adapter to project CLIP text embeddings to desired dimension
    """
    def __init__(self, text_dim: int = 512, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (N, text_dim)
        Returns:
            adapted_features: (N, output_dim)
        """
        return self.adapter(text_features)


class PromptEncoder(nn.Module):
    """
    Learnable prompt tokens that can be prepended to category names
    """
    def __init__(self, num_prompts: int = 4, text_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.num_prompts = num_prompts
        self.text_dim = text_dim
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, text_dim) * 0.02
        )
        
        # Prompt projection
        self.prompt_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Add learnable prompt tokens to text features
        
        Args:
            text_features: (N, text_dim)
        Returns:
            enhanced_features: (N, text_dim)
        """
        batch_size = text_features.size(0)
        
        # Get prompt features
        prompt_features = self.prompt_proj(self.prompt_embeddings) # (num_prompts, text_dim)
        prompt_features = prompt_features.mean(dim=0, keepdim=True) # (1, text_dim)
        prompt_features = prompt_features.expand(batch_size, -1) # (N, text_dim)
        
        # Combine with text features
        enhanced_features = text_features + prompt_features
        enhanced_features = enhanced_features / enhanced_features.norm(dim=-1, keepdim=True)
        
        return enhanced_features


def create_text_encoder(model_name: str = "ViT-B/32", freeze: bool = True, use_adapter: bool = False,
                        adapter_dim: int = 256, use_prompts: bool = False, num_prompts: int = 4,
                        device: str = "cuda") -> nn.Module:
    """
    Create text encoder with optional adapter and prompts
    
    Args:
        model_name: CLIP model name
        freeze: Whether to freeze CLIP weights
        use_adapter: Whether to use text adapter
        adapter_dim: Output dimension of adapter
        use_prompts: Whether to use learnable prompts
        num_prompts: Number of prompt tokens
        device: Device to load model on
    
    Returns:
        text_encoder: Text encoder module
    """
    # Create CLIP encoder
    clip_encoder = CLIPTextEncoder(
        model_name=model_name,
        freeze=freeze,
        device=device
    )
    
    text_dim = clip_encoder.embedding_dim
    
    # Build encoder with optional components
    modules = [clip_encoder]
    
    if use_prompts:
        modules.append(PromptEncoder(
            num_prompts=num_prompts,
            text_dim=text_dim
        ))
    
    if use_adapter:
        modules.append(TextAdapter(
            text_dim=text_dim,
            output_dim=adapter_dim
        ))
    
    if len(modules) == 1:
        return modules[0]
    else:
        return nn.Sequential(*modules)