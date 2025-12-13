"""
COCO Dataset Loader for YOLO-World
Supports text prompts and online vocabulary construction
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools.coco import COCO


class COCODetectionDataset(Dataset):
    """
    COCO Dataset for YOLO-World training with text prompts
    """
    
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 640,
        augment: bool = True,
        cache_images: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self.cache_images = cache_images
        
        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        
        # Category information
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.categories = {cat_id: self.coco.cats[cat_id]['name'] 
                          for cat_id in self.cat_ids}
        
        # All category names for vocabulary
        self.all_category_names = [self.categories[cat_id] for cat_id in self.cat_ids]
        
        print(f"Loaded {len(self.img_ids)} images with {len(self.cat_ids)} categories")
        
        # Image cache
        self.imgs_cache = {} if cache_images else None
        
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, index: int) -> Dict:
        """
        Returns:
            Dictionary with:
            - image: torch.Tensor (3, H, W)
            - boxes: torch.Tensor (N, 4) in xyxy format, normalized [0, 1]
            - labels: torch.Tensor (N,) category indices
            - category_names: List[str] category names present in this image
            - img_id: int
            - original_size: Tuple[int, int] (H, W)
        """
        try:
            img_id = self.img_ids[index]
            
            # Load image
            img_info = self.coco.imgs[img_id]
            img_path = self.img_dir / img_info['file_name']
            
            if self.cache_images and img_id in self.imgs_cache:
                img = self.imgs_cache[img_id].copy()
            else:
                # Use PIL instead of OpenCV for better multiprocessing support
                try:
                    pil_img = Image.open(str(img_path)).convert('RGB')
                    img = np.array(pil_img)  # Convert to numpy array (H, W, 3)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}. Skipping to next image.")
                    # Recursively try the next image
                    return self.__getitem__((index + 1) % len(self.img_ids))
                
                if self.cache_images:
                    self.imgs_cache[img_id] = img.copy()
            
            original_h, original_w = img.shape[:2]
            
            # Load annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Filter valid annotations
            boxes = []
            labels = []
            category_names = []
            
            for ann in anns:
                # Skip crowd annotations
                if ann.get('iscrowd', 0):
                    continue
                    
                bbox = ann['bbox']  # [x, y, w, h]
                cat_id = ann['category_id']
                
                # Convert to xyxy format
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                # Normalize coordinates
                x1 = x1 / original_w
                y1 = y1 / original_h
                x2 = x2 / original_w
                y2 = y2 / original_h
                
                # Filter invalid boxes
                if w > 0 and h > 0 and x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_id_to_idx[cat_id])
                    category_names.append(self.categories[cat_id])
            
            if len(boxes) == 0:
                # Return empty sample if no valid annotations
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = np.array(boxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
            
            # Get unique category names in this image
            unique_category_names = list(set(category_names))
            
            # Apply augmentations and resize
            if self.augment:
                img, boxes, labels = self.augment_hsv(img, boxes, labels)
                img, boxes, labels = self.random_flip(img, boxes, labels)
            
            # Resize to target size
            img, boxes = self.resize(img, boxes, self.img_size)
            
            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            boxes = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()
            
            return {
                'image': img,
                'boxes': boxes,
                'labels': labels,
                'category_names': unique_category_names,
                'img_id': img_id,
                'original_size': (original_h, original_w),
            }
        
        except Exception as e:
            print(f"Error loading image at index {index}: {e}. Skipping to next image.")
            # Recursively try the next image
            return self.__getitem__((index + 1) % len(self.img_ids))
    
    def resize(self, img: np.ndarray, boxes: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and adjust boxes"""
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        dh, dw = target_size - new_h, target_size - new_w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Adjust boxes (currently in normalized [0,1], need to adjust for padding)
        if len(boxes) > 0:
            # Convert to pixel coordinates
            boxes[:, [0, 2]] *= new_w
            boxes[:, [1, 3]] *= new_h
            
            # Add padding offset
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            
            # Normalize back
            boxes /= target_size
            
            # Clip to [0, 1]
            boxes = np.clip(boxes, 0, 1)
        
        return img, boxes
    
    def augment_hsv(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, 
                    hgain=0.015, sgain=0.7, vgain=0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply HSV augmentation"""
        if not self.augment:
            return img, boxes, labels
            
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        
        dtype = img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        return img, boxes, labels
    
    def random_flip(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, 
                    p=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random horizontal flip"""
        if not self.augment or random.random() > p:
            return img, boxes, labels
            
        img = np.fliplr(img).copy()
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        
        return img, boxes, labels
    
    def get_category_names(self) -> List[str]:
        """Return all category names"""
        return self.all_category_names
    
    def get_num_classes(self) -> int:
        """Return number of classes"""
        return len(self.cat_ids)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching
    """
    images = torch.stack([item['image'] for item in batch])
    
    # For boxes and labels, we need to handle variable sizes
    # We'll keep them as lists for now, or pad them
    max_boxes = max([len(item['boxes']) for item in batch])
    
    batch_boxes = []
    batch_labels = []
    batch_num_boxes = []
    
    for item in batch:
        boxes = item['boxes']
        labels = item['labels']
        num_boxes = len(boxes)
        batch_num_boxes.append(num_boxes)
        
        if num_boxes < max_boxes:
            # Pad with zeros
            pad_boxes = torch.zeros((max_boxes - num_boxes, 4), dtype=boxes.dtype)
            pad_labels = torch.zeros((max_boxes - num_boxes,), dtype=labels.dtype) - 1  # -1 for padding
            
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
        
        batch_boxes.append(boxes)
        batch_labels.append(labels)
    
    batch_boxes = torch.stack(batch_boxes)
    batch_labels = torch.stack(batch_labels)
    
    # Collect all category names from the batch
    all_category_names = []
    for item in batch:
        all_category_names.extend(item['category_names'])
    unique_category_names = list(set(all_category_names))
    
    return {
        'images': images,
        'boxes': batch_boxes,
        'labels': batch_labels,
        'num_boxes': torch.tensor(batch_num_boxes, dtype=torch.long),
        'category_names': unique_category_names,
        'img_ids': [item['img_id'] for item in batch],
    }


def create_online_vocabulary(
    batch_categories: List[str],
    all_categories: List[str],
    max_vocab_size: int = 100,
    negative_samples: int = 50,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Create online vocabulary for a batch
    
    Args:
        batch_categories: Categories present in current batch
        all_categories: All available categories
        max_vocab_size: Maximum vocabulary size
        negative_samples: Number of negative categories to sample
    
    Returns:
        vocabulary: List of category names
        category_to_idx: Mapping from category name to index
    """
    # Start with categories in the batch
    vocabulary = list(set(batch_categories))
    
    # Add negative samples
    remaining_categories = list(set(all_categories) - set(vocabulary))
    
    if len(remaining_categories) > 0:
        num_negatives = min(negative_samples, len(remaining_categories))
        negative_cats = random.sample(remaining_categories, num_negatives)
        vocabulary.extend(negative_cats)
    
    # Limit vocabulary size
    if len(vocabulary) > max_vocab_size:
        # Keep all positive categories, sample from negatives
        positive_cats = list(set(batch_categories))
        negative_cats = [c for c in vocabulary if c not in positive_cats]
        
        num_keep_negatives = max_vocab_size - len(positive_cats)
        if num_keep_negatives > 0:
            negative_cats = random.sample(negative_cats, min(num_keep_negatives, len(negative_cats)))
        else:
            negative_cats = []
        
        vocabulary = positive_cats + negative_cats
    
    # Create mapping
    category_to_idx = {cat: idx for idx, cat in enumerate(vocabulary)}
    
    return vocabulary, category_to_idx