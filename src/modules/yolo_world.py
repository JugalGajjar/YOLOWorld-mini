"""
YOLO-World Model: Open-Vocabulary Object Detection
Integrates YOLOv8 backbone with text conditioning and contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from ultralytics.nn.modules import Detect


class ContrastiveDetectionHead(nn.Module):
    """
    Region-text contrastive detection head
    Uses cosine similarity for open-vocabulary classification
    """
    
    def __init__(self, in_channels: List[int] = [256, 512, 512], num_anchors: int = 1,
                 reg_max: int = 16, text_dim: int = 512, temperature: float = 0.01):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.reg_max = reg_max
        self.text_dim = text_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Region embedding heads for each scale
        self.region_embed_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                nn.Conv2d(c, text_dim, 1, bias=False),
            )
            for c in in_channels
        ])
        
        # Box regression heads (similar to YOLO)
        self.box_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                nn.Conv2d(c, 4 * (reg_max + 1), 1), # 4 box coordinates, DFL
            )
            for c in in_channels
        ])
        
        # Objectness heads
        self.obj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                nn.Conv2d(c, 1, 1),
            )
            for c in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor],
                text_embeddings: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: List of feature maps from RepVL-PAN [(B, C, H, W), ...]
            text_embeddings: (N, text_dim) text embeddings for vocabulary
        
        Returns:
            region_embeddings: List of region embeddings per scale
            box_predictions: List of box predictions per scale  
            obj_predictions: List of objectness predictions per scale
        """
        region_embeds_list = []
        box_preds_list = []
        obj_preds_list = []
        
        for i, feat in enumerate(features):
            # Region embeddings
            region_embeds = self.region_embed_heads[i](feat) # (B, text_dim, H, W)
            
            # Normalize region embeddings
            region_embeds = F.normalize(region_embeds, dim=1)
            region_embeds_list.append(region_embeds)
            
            # Box predictions
            box_preds = self.box_heads[i](feat) # (B, 4*(reg_max+1), H, W)
            box_preds_list.append(box_preds)
            
            # Objectness predictions
            obj_preds = self.obj_heads[i](feat) # (B, 1, H, W)
            obj_preds_list.append(obj_preds)
        
        return region_embeds_list, box_preds_list, obj_preds_list
    
    def compute_classification_logits(self, region_embeddings: List[torch.Tensor],
                                      text_embeddings: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute classification logits via cosine similarity
        
        Args:
            region_embeddings: List of (B, text_dim, H, W)
            text_embeddings: (N, text_dim) normalized text embeddings
        
        Returns:
            cls_logits_list: List of (B, N, H, W) classification logits
        """
        cls_logits_list = []
        
        # Normalize text embeddings
        text_embeddings = F.normalize(text_embeddings, dim=1) # (N, text_dim)
        
        for region_embeds in region_embeddings:
            B, D, H, W = region_embeds.shape
            N = text_embeddings.size(0)
            
            # Reshape: (B, D, H, W) -> (B, H, W, D)
            region_embeds = region_embeds.permute(0, 2, 3, 1)
            
            # Compute cosine similarity: region_embeds @ text_embeddings.T
            # (B, H, W, D) @ (D, N) -> (B, H, W, N)
            cls_logits = torch.matmul(region_embeds, text_embeddings.T)
            
            # Scale by temperature
            cls_logits = cls_logits / self.temperature.clamp(min=1e-8)
            
            # Reshape: (B, H, W, N) -> (B, N, H, W)
            cls_logits = cls_logits.permute(0, 3, 1, 2)
            
            cls_logits_list.append(cls_logits)
        
        return cls_logits_list


class YOLOWorld(nn.Module):
    """
    YOLO-World: Open-Vocabulary Object Detector
    """
    
    def __init__(self, yolo_model_path: str, text_encoder: nn.Module, repvl_pan: nn.Module = None,
                 num_classes: int = 80, text_dim: int = 512, reg_max: int = 16, freeze_backbone: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.reg_max = reg_max
        
        # Load YOLOv8 backbone
        yolo = YOLO(yolo_model_path)
        self.backbone = yolo.model.model[:10]
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Detect actual backbone output channels by running a dummy forward pass
        self.backbone_channels = self._detect_backbone_channels()
        print(f"Detected backbone channels: {self.backbone_channels}")
        
        # Text encoder
        self.text_encoder = text_encoder
        
        # Create or use provided RepVL-PAN
        if repvl_pan is None:
            from .repvl_pan import RepVLPAN
            self.repvl_pan = RepVLPAN(
                in_channels=self.backbone_channels,
                text_dim=text_dim,
                use_attention=True,
                num_heads=8
            )
        else:
            self.repvl_pan = repvl_pan
        
        # Detection head
        self.detection_head = ContrastiveDetectionHead(
            in_channels=self.backbone_channels,
            text_dim=text_dim,
            reg_max=reg_max
        )
    
    def _detect_backbone_channels(self):
        """Detect backbone output channels by running a dummy forward pass"""
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Move to same device as backbone
        device = next(self.backbone.parameters()).device
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            features = self.extract_backbone_features(dummy_input)
        
        channels = [f.shape[1] for f in features]
        return channels
    
    def extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from backbone
        
        Args:
            x: (B, 3, H, W)
        Returns:
            features: List of feature maps at different scales
        """
        features = []
        
        # YOLOv8 backbone structure
        for i, module in enumerate(self.backbone):
            x = module(x)
            
            # Collect features at specific layers
            # For YOLOv8-S -> 6, 8, 9
            if i in [6, 8, 9]:
                features.append(x)
        
        if len(features) != 3:
            print(f"Warning: Expected 3 feature maps, got {len(features)}. Adjusting...")
            # If we have more, take the last 3
            if len(features) > 3:
                features = features[-3:]
            # If we have less, this is a problem - use what we have
            elif len(features) == 0:
                # Fallback: just use the final output at 3 scales (duplicate)
                features = [x, x, x]
                print("Warning: No features extracted, using final output")
        
        return features
    
    def forward(self, images: torch.Tensor, text_embeddings: Optional[torch.Tensor] = None,
                category_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: (B, 3, H, W)
            text_embeddings: (N, text_dim) precomputed text embeddings
            category_names: List of category names (if text_embeddings not provided)
        
        Returns:
            outputs: Dictionary with predictions
        """
        # Encode text if not provided
        if text_embeddings is None:
            assert category_names is not None, "Must provide either text_embeddings or category_names"
            text_embeddings = self.text_encoder.encode_text(category_names)
        
        # Extract backbone features
        backbone_features = self.extract_backbone_features(images)
        
        # Apply RepVL-PAN
        enhanced_features = self.repvl_pan(backbone_features, text_embeddings)
        
        # Detection head
        region_embeds, box_preds, obj_preds = self.detection_head(
            enhanced_features, text_embeddings
        )
        
        # Compute classification logits
        cls_logits = self.detection_head.compute_classification_logits(
            region_embeds, text_embeddings
        )
        
        return {
            'cls_logits': cls_logits,
            'box_preds': box_preds,
            'obj_preds': obj_preds,
            'region_embeds': region_embeds,
            'text_embeddings': text_embeddings
        }
    
    def predict(self, images: torch.Tensor, category_names: List[str], conf_threshold: float = 0.25,
                iou_threshold: float = 0.7, max_det: int = 300) -> List[Dict]:
        """
        Inference with NMS
        
        Args:
            images: (B, 3, H, W)
            category_names: List of category names for open vocabulary
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections per image
        
        Returns:
            predictions: List of dictionaries with detections per image
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(images, category_names=category_names)
            
            # Post-process predictions
            predictions = self.post_process(
                outputs, 
                conf_threshold, 
                iou_threshold, 
                max_det,
                category_names
            )
        
        return predictions
    
    def post_process(self, outputs: Dict, conf_threshold: float, iou_threshold: float, max_det: int,
                     category_names: List[str]) -> List[Dict]:
        """
        Post-process predictions with NMS
        """
        from modules.losses import decode_boxes, generate_anchors
        from utils.utils import box_nms
        
        cls_logits = outputs['cls_logits']
        box_preds = outputs['box_preds']
        obj_preds = outputs['obj_preds']
        
        batch_size = cls_logits[0].size(0)
        device = cls_logits[0].device
        strides = [8, 16, 32]
        
        predictions = []
        
        for b in range(batch_size):
            # Collect predictions from all scales
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for scale_idx in range(len(cls_logits)):
                cls_logit = cls_logits[scale_idx][b] # (N_vocab, H, W)
                box_pred = box_preds[scale_idx][b:b+1] # (1, 4*(reg_max+1), H, W)
                obj_pred = obj_preds[scale_idx][b] # (1, H, W)
                
                N_vocab, H, W = cls_logit.shape
                stride = strides[scale_idx]
                
                # Generate anchors
                anchors = generate_anchors((H, W), stride, device) # (H*W, 2)
                
                # Decode boxes
                decoded_boxes = decode_boxes(
                    box_pred, anchors, stride, self.reg_max
                )  # (1, H*W, 4)
                decoded_boxes = decoded_boxes.squeeze(0) # (H*W, 4)
                
                # Normalize boxes to [0, 1]
                img_size = 640 # Default image size
                decoded_boxes = decoded_boxes / img_size
                decoded_boxes = decoded_boxes.clamp(0, 1)
                
                # Reshape decoded boxes to (H, W, 4)
                decoded_boxes = decoded_boxes.reshape(H, W, 4)
                
                # Apply sigmoid to get probabilities
                obj_scores = torch.sigmoid(obj_pred).squeeze(0) # (H, W)
                cls_scores = torch.sigmoid(cls_logit) # (N_vocab, H, W)
                
                # Combine objectness and class scores
                scores = obj_scores.unsqueeze(0) * cls_scores # (N_vocab, H, W)
                
                # Get max class and score for each location
                max_scores, max_labels = scores.max(dim=0) # (H, W)
                
                # Filter by confidence
                mask = max_scores > conf_threshold
                
                if mask.sum() == 0:
                    continue
                
                # Get filtered predictions
                filtered_boxes = decoded_boxes[mask] # (N_filtered, 4)
                filtered_scores = max_scores[mask] # (N_filtered,)
                filtered_labels = max_labels[mask] # (N_filtered,)
                
                all_boxes.append(filtered_boxes)
                all_scores.append(filtered_scores)
                all_labels.append(filtered_labels)
            
            # Combine all scales
            if len(all_boxes) > 0:
                all_boxes = torch.cat(all_boxes, dim=0) # (N_total, 4)
                all_scores = torch.cat(all_scores, dim=0) # (N_total,)
                all_labels = torch.cat(all_labels, dim=0) # (N_total,)
                
                # Apply NMS
                keep_indices = box_nms(all_boxes, all_scores, iou_threshold)
                
                # Limit to max_det
                if len(keep_indices) > max_det:
                    # Sort by score and keep top max_det
                    scores_sorted, sort_indices = all_scores[keep_indices].sort(descending=True)
                    keep_indices = keep_indices[sort_indices[:max_det]]
                
                # Get final predictions
                final_boxes = all_boxes[keep_indices]
                final_scores = all_scores[keep_indices]
                final_labels = all_labels[keep_indices]
                
                # Convert label indices to category names
                final_category_names = [category_names[int(label)] for label in final_labels]
                
                predictions.append({
                    'boxes': final_boxes.cpu(),
                    'scores': final_scores.cpu(),
                    'labels': final_labels.cpu(),
                    'category_names': final_category_names
                })
            else:
                # No detections
                predictions.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros((0,)),
                    'labels': torch.zeros((0,), dtype=torch.long),
                    'category_names': []
                })
        
        return predictions


def build_yolo_world(config: Dict, device: str = "cuda") -> YOLOWorld:
    """
    Build YOLO-World model from configuration
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        model: YOLOWorld model
    """
    from .text_encoder import create_text_encoder
    
    # Create text encoder
    text_encoder = create_text_encoder(
        model_name=config['model']['clip_model'],
        freeze=config['model']['freeze_clip'],
        device=device
    )
    
    # Create YOLO-World model (it will auto-detect backbone channels and create RepVL-PAN)
    model = YOLOWorld(
        yolo_model_path=config['model']['pretrained_yolo'],
        text_encoder=text_encoder,
        repvl_pan=None,  # Let model create it after detecting channels
        num_classes=config['data']['num_classes'],
        text_dim=config['model']['text_dim']
    )
    
    return model.to(device)