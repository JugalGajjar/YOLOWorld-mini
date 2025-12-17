"""
Loss Functions for YOLO-World
Complete implementation with target assignment, box regression, and contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format
    
    Returns:
        iou: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0) # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1] # (N, M)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / (union + 1e-7)
    
    return iou


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride: int,
                 reg_max: int = 16) -> torch.Tensor:
    """
    Decode box predictions from DFL format to xyxy boxes
    
    Args:
        box_preds: (B, 4*(reg_max+1), H, W) or (N, 4*(reg_max+1)) box predictions in DFL format
        anchors: (H*W, 2) or (N, 2) anchor centers in absolute coordinates
        stride: Feature map stride
        reg_max: Maximum regression value
    
    Returns:
        boxes: (B, H*W, 4) or (N, 4) decoded boxes in xyxy format (normalized)
    """
    # Handle different input shapes
    if box_preds.dim() == 4:
        B, _, H, W = box_preds.shape
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().reshape(B, H*W, 4*(reg_max+1))
        batch_mode = True
    else:
        box_preds = box_preds.contiguous().reshape(-1, 4*(reg_max+1))
        batch_mode = False
    
    # Split into ltrb predictions
    box_preds = box_preds.reshape(*box_preds.shape[:-1], 4, reg_max+1) # (..., 4, reg_max+1)
    
    # Apply softmax along the reg_max dimension to get distribution
    box_preds = F.softmax(box_preds, dim=-1)
    
    # Create projection vector
    device = box_preds.device
    proj = torch.arange(reg_max + 1, device=device, dtype=torch.float32)
    
    # Compute expected value: sum(prob * value)
    box_preds = (box_preds * proj.reshape(1, 1, 1, -1)).sum(dim=-1) # (..., 4)
    
    # Scale by stride
    box_preds = box_preds * stride
    
    # Expand anchors for batch
    if batch_mode:
        B = box_preds.shape[0]
        anchors = anchors.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
    
    # Decode: ltrb to xyxy
    # anchors: (B, H*W, 2) or (N, 2) with [x_center, y_center]
    # box_preds: (B, H*W, 4) or (N, 4) with [left, top, right, bottom]
    x1 = anchors[..., 0] - box_preds[..., 0] # x_center - left
    y1 = anchors[..., 1] - box_preds[..., 1] # y_center - top
    x2 = anchors[..., 0] + box_preds[..., 2] # x_center + right
    y2 = anchors[..., 1] + box_preds[..., 3] # y_center + bottom
    
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    return boxes


def generate_anchors(featmap_size: Tuple[int, int], stride: int, device: torch.device) -> torch.Tensor:
    """
    Generate anchor points for a feature map
    
    Args:
        featmap_size: (H, W) feature map size
        stride: Stride of the feature map
        device: Device to create anchors on
    
    Returns:
        anchors: (H*W, 2) anchor centers in absolute coordinates
    """
    H, W = featmap_size
    
    # Create grid
    shift_x = torch.arange(0, W, device=device) * stride + stride // 2
    shift_y = torch.arange(0, H, device=device) * stride + stride // 2
    
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    
    anchors = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
    
    return anchors


class TargetAssigner:
    """
    Assigns ground truth targets to predictions using IoU matching
    """
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5, center_radius: float = 2.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.center_radius = center_radius
    
    def assign_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, num_boxes: int,
                       img_size: int = 640) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assign ground truth to anchors
        
        Args:
            anchors: (N_anchors, 2) anchor centers
            gt_boxes: (max_boxes, 4) ground truth boxes in normalized xyxy
            gt_labels: (max_boxes,) ground truth labels
            num_boxes: Number of valid boxes
            img_size: Image size for denormalization
        
        Returns:
            assigned_labels: (N_anchors,) assigned class labels (-1 for ignore, num_classes for background)
            assigned_boxes: (N_anchors, 4) assigned box targets
            positive_mask: (N_anchors,) mask for positive samples
            matched_gt_idx: (N_anchors,) indices of matched ground truth
        """
        num_anchors = anchors.shape[0]
        device = anchors.device
        
        # Initialize assignments
        assigned_labels = torch.full((num_anchors,), self.num_classes, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_anchors, 4), device=device)
        positive_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        matched_gt_idx = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        
        if num_boxes == 0:
            return assigned_labels, assigned_boxes, positive_mask, matched_gt_idx
        
        # Get valid ground truth
        gt_boxes_valid = gt_boxes[:num_boxes] # (N_gt, 4)
        gt_labels_valid = gt_labels[:num_boxes] # (N_gt,)
        
        # Denormalize boxes to absolute coordinates
        gt_boxes_abs = gt_boxes_valid.clone()
        gt_boxes_abs[:, [0, 2]] *= img_size
        gt_boxes_abs[:, [1, 3]] *= img_size
        
        # Create anchor boxes (point-based)
        anchor_size = 8.0 # Small fixed size for matching
        anchor_boxes = torch.zeros((num_anchors, 4), device=device)
        anchor_boxes[:, 0] = anchors[:, 0] - anchor_size / 2
        anchor_boxes[:, 1] = anchors[:, 1] - anchor_size / 2
        anchor_boxes[:, 2] = anchors[:, 0] + anchor_size / 2
        anchor_boxes[:, 3] = anchors[:, 1] + anchor_size / 2
        
        # Check if anchors are inside ground truth boxes (center sampling)
        is_in_boxes = []
        for gt_box in gt_boxes_abs:
            l = anchors[:, 0] - gt_box[0]
            t = anchors[:, 1] - gt_box[1]
            r = gt_box[2] - anchors[:, 0]
            b = gt_box[3] - anchors[:, 1]
            
            is_in_box = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0
            is_in_boxes.append(is_in_box)
        
        is_in_boxes = torch.stack(is_in_boxes, dim=1) # (N_anchors, N_gt)
        
        # Check center region (stricter constraint)
        gt_centers = (gt_boxes_abs[:, :2] + gt_boxes_abs[:, 2:]) / 2
        gt_wh = gt_boxes_abs[:, 2:] - gt_boxes_abs[:, :2]
        
        is_in_centers = []
        for i, (center, wh) in enumerate(zip(gt_centers, gt_wh)):
            # Center region is a smaller box
            radius = self.center_radius
            center_box = torch.stack([
                center[0] - radius * wh[0] / 2,
                center[1] - radius * wh[1] / 2,
                center[0] + radius * wh[0] / 2,
                center[1] + radius * wh[1] / 2
            ])
            
            l = anchors[:, 0] - center_box[0]
            t = anchors[:, 1] - center_box[1]
            r = center_box[2] - anchors[:, 0]
            b = center_box[3] - anchors[:, 1]
            
            is_in_center = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0
            is_in_centers.append(is_in_center)
        
        is_in_centers = torch.stack(is_in_centers, dim=1) # (N_anchors, N_gt)
        
        # Combine constraints
        is_in_gts = is_in_boxes & is_in_centers
        
        # For each ground truth, find candidate anchors
        for gt_idx in range(num_boxes):
            candidate_mask = is_in_gts[:, gt_idx]
            
            if candidate_mask.sum() == 0:
                # If no candidates, use closest anchor
                distances = torch.sqrt(
                    (anchors[:, 0] - gt_centers[gt_idx, 0]) ** 2 +
                    (anchors[:, 1] - gt_centers[gt_idx, 1]) ** 2
                )
                closest_idx = distances.argmin()
                candidate_mask[closest_idx] = True
            
            # Assign to candidates
            assigned_labels[candidate_mask] = gt_labels_valid[gt_idx]
            assigned_boxes[candidate_mask] = gt_boxes_valid[gt_idx]
            positive_mask[candidate_mask] = True
            matched_gt_idx[candidate_mask] = gt_idx
        
        return assigned_labels, assigned_boxes, positive_mask, matched_gt_idx


class ContrastiveLoss(nn.Module):
    """
    Region-Text Contrastive Loss
    Aligns region embeddings with corresponding text embeddings
    """
    
    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, region_embeddings: torch.Tensor, text_embeddings: torch.Tensor, labels: torch.Tensor,
                positive_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_embeddings: (N_regions, D) normalized region embeddings
            text_embeddings: (N_vocab, D) normalized text embeddings  
            labels: (N_regions,) label indices for each region
            positive_mask: (N_regions,) mask indicating valid regions
        
        Returns:
            loss: Scalar contrastive loss
        """
        # Filter valid regions
        valid_mask = positive_mask.bool()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=region_embeddings.device, requires_grad=True)
        
        region_embeddings = region_embeddings[valid_mask]
        labels = labels[valid_mask]
        
        # Normalize embeddings
        region_embeddings = F.normalize(region_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix: (N_valid, N_vocab)
        similarity = torch.matmul(region_embeddings, text_embeddings.T)
        similarity = similarity / self.temperature
        
        # Create one-hot labels
        num_regions = region_embeddings.size(0)
        num_classes = text_embeddings.size(0)
        
        # Handle labels that might be out of vocabulary
        valid_label_mask = (labels >= 0) & (labels < num_classes)
        
        if valid_label_mask.sum() == 0:
            return torch.tensor(0.0, device=region_embeddings.device, requires_grad=True)
        
        similarity = similarity[valid_label_mask]
        labels = labels[valid_label_mask]
        
        # Cross-entropy loss (InfoNCE loss)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N,) predicted logits
            target: (N,) binary targets (0 or 1)
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal weight
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Alpha weighting
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


class BoxLoss(nn.Module):
    """
    Bounding Box Loss (CIoU-based)
    """
    def __init__(self, loss_type: str = 'ciou', eps: float = 1e-7):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) in xyxy format, normalized [0, 1]
            target_boxes: (N, 4) in xyxy format, normalized [0, 1]
        """
        if pred_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        if self.loss_type == 'giou':
            loss = self.giou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'ciou':
            loss = self.ciou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'iou':
            loss = self.iou_loss(pred_boxes, target_boxes)
        else:
            loss = F.l1_loss(pred_boxes, target_boxes, reduction='none').sum(dim=1)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """IoU loss"""
        inter = self.intersection(pred, target)
        union = self.box_area(pred) + self.box_area(target) - inter
        iou = inter / (union + self.eps)
        return 1 - iou
    
    def giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generalized IoU loss"""
        # IoU
        inter = self.intersection(pred, target)
        union = self.box_area(pred) + self.box_area(target) - inter
        iou = inter / (union + self.eps)
        
        # Enclosing box
        x1 = torch.min(pred[:, 0], target[:, 0])
        y1 = torch.min(pred[:, 1], target[:, 1])
        x2 = torch.max(pred[:, 2], target[:, 2])
        y2 = torch.max(pred[:, 3], target[:, 3])
        
        enclosing_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        # GIoU
        giou = iou - (enclosing_area - union) / (enclosing_area + self.eps)
        
        return 1 - giou
    
    def ciou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Complete IoU loss"""
        # IoU
        inter = self.intersection(pred, target)
        union = self.box_area(pred) + self.box_area(target) - inter
        iou = inter / (union + self.eps)
        
        # Center distance
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        center_dist_sq = ((pred_center - target_center) ** 2).sum(dim=1)
        
        # Diagonal of enclosing box
        x1 = torch.min(pred[:, 0], target[:, 0])
        y1 = torch.min(pred[:, 1], target[:, 1])
        x2 = torch.max(pred[:, 2], target[:, 2])
        y2 = torch.max(pred[:, 3], target[:, 3])
        diagonal_sq = (x2 - x1).clamp(min=0) ** 2 + (y2 - y1).clamp(min=0) ** 2
        
        # Aspect ratio consistency
        pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=self.eps)
        pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=self.eps)
        target_w = (target[:, 2] - target[:, 0]).clamp(min=self.eps)
        target_h = (target[:, 3] - target[:, 1]).clamp(min=self.eps)
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - (center_dist_sq / (diagonal_sq + self.eps) + alpha * v)
        
        return 1 - ciou
    
    @staticmethod
    def intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute intersection area"""
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        return inter
    
    @staticmethod
    def box_area(boxes: torch.Tensor) -> torch.Tensor:
        """Compute box area"""
        return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


class DFLLoss(nn.Module):
    """
    Distribution Focal Loss for box regression
    Used in YOLOv8 for learning box distributions
    """
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, 4*(reg_max+1)) predicted distributions
            target: (N, 4) target coordinates
        """
        # Convert target to distribution format
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left
        
        # Compute loss for each coordinate
        pred = pred.contiguous().reshape(-1, 4, self.reg_max + 1)
        
        loss = 0
        for i in range(4):
            # Left distribution
            loss_left = F.cross_entropy(
                pred[:, i, :],
                target_left[:, i],
                reduction='none'
            ) * weight_left[:, i]
            
            # Right distribution  
            loss_right = F.cross_entropy(
                pred[:, i, :],
                target_right[:, i],
                reduction='none'
            ) * weight_right[:, i]
            
            loss += (loss_left + loss_right).mean()
        
        return loss / 4


class YOLOWorldLoss(nn.Module):
    """
    Complete YOLO-World Loss with proper target assignment and box decoding
    """
    def __init__(self, num_classes: int = 80, box_loss_weight: float = 7.5, cls_loss_weight: float = 0.5,
                 obj_loss_weight: float = 1.0, contrastive_loss_weight: float = 1.0, img_size: int = 640,
                 reg_max: int = 16):
        super().__init__()
        
        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.img_size = img_size
        self.reg_max = reg_max
        
        # Loss functions
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.01)
        self.box_loss_fn = BoxLoss(loss_type='ciou')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Target assigner
        self.target_assigner = TargetAssigner(num_classes=num_classes)
        
        # Feature map strides
        self.strides = [8, 16, 32]
    
    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss with proper target assignment and box decoding
        
        Args:
            outputs: Model outputs with cls_logits, box_preds, obj_preds, region_embeds, text_embeddings
            targets: Ground truth with boxes, labels, num_boxes
        
        Returns:
            total_loss: Total loss scalar
            loss_dict: Dictionary of individual losses
        """
        # Extract outputs
        cls_logits = outputs['cls_logits'] # List of (B, N_vocab, H, W) per scale
        box_preds = outputs['box_preds'] # List of (B, 4*(reg_max+1), H, W) per scale
        obj_preds = outputs['obj_preds'] # List of (B, 1, H, W) per scale
        region_embeds = outputs['region_embeds'] # List of (B, D, H, W) per scale
        text_embeddings = outputs['text_embeddings'] # (N_vocab, D)
        
        # Extract targets
        target_boxes = targets['boxes'] # (B, max_boxes, 4) in normalized xyxy
        target_labels = targets['labels'] # (B, max_boxes)
        num_boxes = targets['num_boxes'] # (B,)
        label_to_vocab = targets.get('label_to_vocab', None) # (num_classes,) mapping
        
        batch_size = target_boxes.shape[0]
        device = target_boxes.device
        
        # Initialize losses
        total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        num_positive = 0
        
        # Process each scale
        all_region_embeds = []
        all_assigned_labels = []
        all_positive_masks = []
        
        for scale_idx, stride in enumerate(self.strides):
            cls_logit = cls_logits[scale_idx] # (B, N_vocab, H, W)
            box_pred = box_preds[scale_idx] # (B, 4*(reg_max+1), H, W)
            obj_pred = obj_preds[scale_idx] # (B, 1, H, W)
            region_embed = region_embeds[scale_idx] # (B, D, H, W)
            
            B, N_vocab, H, W = cls_logit.shape
            _, D, _, _ = region_embed.shape
            
            # Generate anchors for this scale
            anchors = generate_anchors((H, W), stride, device) # (H*W, 2)
            
            # Decode box predictions to xyxy format
            decoded_boxes = decode_boxes(
                box_pred, anchors, stride, self.reg_max
            )  # (B, H*W, 4) in absolute coordinates
            
            # Normalize decoded boxes
            decoded_boxes = decoded_boxes / self.img_size # Normalize to [0, 1]
            decoded_boxes = decoded_boxes.clamp(0, 1)
            
            # Flatten predictions
            cls_logit_flat = cls_logit.permute(0, 2, 3, 1).reshape(B, H*W, N_vocab)
            obj_pred_flat = obj_pred.permute(0, 2, 3, 1).reshape(B, H*W, 1)
            region_embed_flat = region_embed.permute(0, 2, 3, 1).reshape(B, H*W, D)
            
            # Process each image in batch
            for b in range(B):
                # Assign targets for this image
                assigned_labels, assigned_boxes, positive_mask, _ = self.target_assigner.assign_targets(
                    anchors=anchors,
                    gt_boxes=target_boxes[b],
                    gt_labels=target_labels[b],
                    num_boxes=num_boxes[b].item(),
                    img_size=self.img_size
                )
                
                # Objectness targets (1 for positive, 0 for background)
                obj_targets = positive_mask.float().unsqueeze(-1)  # (H*W, 1)
                
                # Objectness loss
                obj_loss = self.bce_loss(obj_pred_flat[b], obj_targets).mean()
                total_obj_loss = total_obj_loss + obj_loss
                
                # Classification and box loss (only for positive samples)
                if positive_mask.sum() > 0:
                    # Classification loss
                    pos_cls_logits = cls_logit_flat[b][positive_mask] # (N_pos, N_vocab)
                    pos_labels = assigned_labels[positive_mask] # (N_pos,)
                    
                    # Remap labels to vocabulary indices
                    if label_to_vocab is not None:
                        pos_labels = label_to_vocab[pos_labels]
                        # Filter out unmapped labels (-1)
                        valid_mask = pos_labels >= 0
                        pos_cls_logits = pos_cls_logits[valid_mask]
                        pos_labels = pos_labels[valid_mask]
                        pos_target_boxes_orig = assigned_boxes[positive_mask]
                        pos_decoded_boxes_orig = decoded_boxes[b][positive_mask]
                    else:
                        valid_mask = torch.ones(pos_labels.size(0), dtype=torch.bool, device=pos_labels.device)
                        pos_target_boxes_orig = assigned_boxes[positive_mask]
                        pos_decoded_boxes_orig = decoded_boxes[b][positive_mask]
                    
                    # Only compute losses if we have valid labels
                    if valid_mask.sum() > 0:
                        # Clamp labels to valid vocabulary range
                        pos_labels = pos_labels.clamp(0, N_vocab - 1)
                        
                        # Create one-hot targets
                        cls_targets = F.one_hot(pos_labels, num_classes=N_vocab).float()
                        
                        # Classification loss
                        cls_loss = self.bce_loss(pos_cls_logits, cls_targets).sum() / (pos_labels.size(0) + 1e-7)
                        total_cls_loss = total_cls_loss + cls_loss
                        
                        # Box loss with filtered boxes
                        pos_decoded_boxes = pos_decoded_boxes_orig[valid_mask]
                        pos_target_boxes = pos_target_boxes_orig[valid_mask]
                        
                        # Compute CIoU loss
                        box_loss = self.box_loss_fn(pos_decoded_boxes, pos_target_boxes)
                        total_box_loss = total_box_loss + box_loss
                        
                        num_positive += pos_decoded_boxes.size(0)
                
                # Collect for contrastive loss
                all_region_embeds.append(region_embed_flat[b])
                all_assigned_labels.append(assigned_labels)
                all_positive_masks.append(positive_mask)
        
        # Contrastive loss
        if num_positive > 0 and len(all_region_embeds) > 0:
            all_region_embeds = torch.cat(all_region_embeds, dim=0) # (B*H*W, D)
            all_assigned_labels = torch.cat(all_assigned_labels, dim=0) # (B*H*W,)
            all_positive_masks = torch.cat(all_positive_masks, dim=0) # (B*H*W,)
            
            contrastive_loss = self.contrastive_loss_fn(
                all_region_embeds,
                text_embeddings,
                all_assigned_labels,
                all_positive_masks
            )
            total_contrastive_loss = total_contrastive_loss + contrastive_loss
        
        # Average over scales and batch
        num_scales = len(self.strides)
        total_cls_loss = total_cls_loss / (batch_size * num_scales)
        total_obj_loss = total_obj_loss / (batch_size * num_scales)
        total_box_loss = total_box_loss / max(num_positive, 1)
        
        # Combined loss
        total_loss = (
            self.cls_loss_weight * total_cls_loss +
            self.obj_loss_weight * total_obj_loss +
            self.box_loss_weight * total_box_loss +
            self.contrastive_loss_weight * total_contrastive_loss
        )
        
        loss_dict = {
            'loss': total_loss.item(),
            'cls_loss': total_cls_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'box_loss': total_box_loss.item(),
            'contrastive_loss': total_contrastive_loss.item(),
            'num_positive': num_positive
        }
        
        return total_loss, loss_dict