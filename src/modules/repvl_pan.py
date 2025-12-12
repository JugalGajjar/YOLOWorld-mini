"""
RepVL-PAN: Text-Aware Feature Pyramid Network for YOLO-World
Integrates text conditioning into multi-scale features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class TextGuidedConv(nn.Module):
    """
    Text-guided convolutional block that modulates features based on text embeddings
    """
    
    def __init__(self, in_channels: int, out_channels: int, text_dim: int = 512, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Text modulation: generates scale and bias from text
        self.text_modulation = nn.Sequential(
            nn.Linear(text_dim, out_channels * 2),
            nn.LayerNorm(out_channels * 2)
        )
        
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual features (B, C, H, W)
            text_features: Text embeddings (B, text_dim) or (N, text_dim)
        Returns:
            modulated_features: (B, C, H, W)
        """
        # Standard convolution
        out = self.conv(x)
        out = self.bn(out)
        
        # Text modulation
        if text_features.dim() == 2:
            # If text_features is (N, text_dim), pool to (B, text_dim)
            if text_features.size(0) != x.size(0):
                # Average over all text embeddings
                text_features = text_features.mean(dim=0, keepdim=True)
                text_features = text_features.expand(x.size(0), -1)
        
        # Generate scale and bias
        modulation = self.text_modulation(text_features) # (B, 2*C)
        scale, bias = modulation.chunk(2, dim=1) # (B, C), (B, C)
        
        # Apply modulation: scale * features + bias
        scale = scale.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        bias = bias.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        
        out = out * (1 + scale) + bias
        out = self.act(out)
        
        return out


class ImageTextAttention(nn.Module):
    """
    Cross-attention between image features and text features
    """
    
    def __init__(self, img_channels: int, text_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = img_channels // num_heads
        assert img_channels % num_heads == 0, "img_channels must be divisible by num_heads"
        
        # Query from image, Key and Value from text
        self.q_proj = nn.Linear(img_channels, img_channels)
        self.k_proj = nn.Linear(text_dim, img_channels)
        self.v_proj = nn.Linear(text_dim, img_channels)
        
        self.out_proj = nn.Linear(img_channels, img_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(img_channels)
    
    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features: (B, C, H, W)
            text_features: (N, text_dim) where N is number of text prompts
        Returns:
            attended_features: (B, C, H, W)
        """
        B, C, H, W = img_features.shape
        N = text_features.size(0)
        
        # Reshape image features: (B, C, H, W) -> (B, H*W, C)
        img_flat = img_features.flatten(2).permute(0, 2, 1) # (B, H*W, C)
        
        # Expand text features for batch
        text_features = text_features.unsqueeze(0).expand(B, -1, -1) # (B, N, text_dim)
        
        # Compute Q, K, V
        Q = self.q_proj(img_flat) # (B, H*W, C)
        K = self.k_proj(text_features) # (B, N, C)
        V = self.v_proj(text_features) # (B, N, C)
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, H*W, head_dim)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        
        # Attention: Q @ K.T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, num_heads, H*W, N)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V) # (B, num_heads, H*W, head_dim)
        
        # Reshape back 
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, H*W, C)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and norm
        output = self.norm(output + img_flat)
        
        # Reshape back to image format
        output = output.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        
        return output


class RepVLBlock(nn.Module):
    """
    Single RepVL block with text conditioning
    """
    
    def __init__(self, in_channels: int, out_channels: int, text_dim: int = 512, use_attention: bool = True,
                 num_heads: int = 8):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Text-guided convolution
        self.text_conv = TextGuidedConv(
            in_channels, out_channels, text_dim
        )
        
        # Optional cross-attention
        if use_attention:
            self.attention = ImageTextAttention(
                out_channels, text_dim, num_heads
            )
        
        # Standard convolution for residual path
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            text_features: (N, text_dim)
        Returns:
            output: (B, C_out, H, W)
        """
        identity = self.shortcut(x)
        
        # Text-guided convolution
        out = self.text_conv(x, text_features)
        
        # Cross-attention
        if self.use_attention:
            out = self.attention(out, text_features)
        
        # Residual
        out = out + identity
        
        return out


class RepVLPAN(nn.Module):
    """
    RepVL-PAN: Text-aware Path Aggregation Network
    Modified PAN neck with text conditioning at multiple scales
    """
    
    def __init__(self, in_channels: List[int] = [256, 512, 512], # From YOLOv8 backbone
                 text_dim: int = 512, use_attention: bool = True, num_heads: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.text_dim = text_dim
        
        # Upsampling path (top-down)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # RepVL blocks for each scale
        self.repvl_blocks = nn.ModuleList([
            RepVLBlock(
                in_channels[i], in_channels[i], text_dim, 
                use_attention, num_heads
            )
            for i in range(len(in_channels))
        ])
        
        # Channel adaptation layers for fusion
        self.channel_adapters = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            # Adapt from in_channels[i+1] to in_channels[i]
            if in_channels[i+1] != in_channels[i]:
                self.channel_adapters.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[i+1], in_channels[i], 1, bias=False),
                        nn.BatchNorm2d(in_channels[i])
                    )
                )
            else:
                self.channel_adapters.append(nn.Identity())
        
        # Fusion convolutions
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], in_channels[i], 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channels[i]),
                nn.SiLU(inplace=True)
            )
            for i in range(len(in_channels))
        ])
    
    def forward(self, features: List[torch.Tensor], text_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            features: List of multi-scale features from backbone
                      [(B, C0, H0, W0), (B, C1, H1, W1), (B, C2, H2, W2)]
                      where H0 > H1 > H2 (decreasing spatial resolution)
            text_features: (N, text_dim) text embeddings
        
        Returns:
            enhanced_features: List of text-enhanced features at same scales
        """
        assert len(features) == len(self.in_channels), \
            f"Expected {len(self.in_channels)} features, got {len(features)}"
        
        # Apply RepVL blocks to each scale
        repvl_features = []
        for i, (feat, block) in enumerate(zip(features, self.repvl_blocks)):
            enhanced = block(feat, text_features)
            repvl_features.append(enhanced)
        
        # Top-down path aggregation (similar to FPN)
        outputs = []
        
        # Starting from the smallest feature map (last one)
        x = repvl_features[-1]
        outputs.insert(0, self.fusion_convs[-1](x))
        
        # Upsample and fuse
        for i in range(len(features) - 2, -1, -1):
            # Upsample
            x_up = self.upsample(x)
            
            # Adapt channels
            x_up = self.channel_adapters[i](x_up)
            
            # Match spatial dimensions
            if x_up.shape[2:] != repvl_features[i].shape[2:]:
                x_up = F.interpolate(
                    x_up, size=repvl_features[i].shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Add
            x = x_up + repvl_features[i]
            x = self.fusion_convs[i](x)
            outputs.insert(0, x)
        
        return outputs


class SimpleRepVLPAN(nn.Module):
    """
    Simplified RepVL-PAN for faster prototyping
    Only applies text conditioning without complex attention
    """
    
    def __init__(self, in_channels: List[int] = [256, 512, 512], text_dim: int = 512):
        super().__init__()
        
        self.in_channels = in_channels
        
        # Text-guided convolutions for each scale
        self.text_convs = nn.ModuleList([
            TextGuidedConv(c, c, text_dim)
            for c in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor], text_features: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            features: List of multi-scale features
            text_features: (N, text_dim)
        Returns:
            enhanced_features: List of text-enhanced features
        """
        outputs = []
        for feat, text_conv in zip(features, self.text_convs):
            enhanced = text_conv(feat, text_features)
            outputs.append(enhanced)
        
        return outputs