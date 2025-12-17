"""
Utility Functions for YOLO-World
Includes config loading, logging, metrics, and other helpers
"""

import yaml
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logger(log_dir: str, name: str = 'yolo_world') -> logging.Logger:
    """Setup logger for training"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'train_{timestamp}.log')
    )
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str = 'checkpoint.pth', is_best: bool = False):
    """Save training checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler: Any = None) -> Dict[str, Any]:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


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


def box_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while sorted_indices.numel() > 0:
        # Keep highest scoring box
        idx = sorted_indices[0]
        keep.append(idx.item()) # <- Convert to Python int
        
        if sorted_indices.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[idx:idx+1], boxes[sorted_indices[1:]])[0]
        
        # Remove boxes with high IoU (keep boxes with LOW overlap)
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xyxy to xywh format"""
    xywh = boxes.clone()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0] # width
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1] # height
    return xywh


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xywh to xyxy format"""
    xyxy = boxes.clone()
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] # y2
    return xyxy


def get_world_size() -> int:
    """Get number of distributed processes"""
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Get rank of current process"""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    """Check if current process is main process"""
    return get_rank() == 0


def synchronize():
    """Synchronize all processes"""
    if get_world_size() > 1:
        torch.distributed.barrier()


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy/mAP
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current validation score
        
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def clip_gradients(model: torch.nn.Module, max_norm: float = 10.0):
    """Clip gradients by norm"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)