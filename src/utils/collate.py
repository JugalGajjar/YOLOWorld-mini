"""
Collate function for COCO-style batches with variable number of boxes
"""

from typing import List, Dict, Any

import torch


def coco_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for COCO-style batches with variable number of boxes.
    Returns:
      {
        "images": (B, 3, H, W),
        "boxes": list[Tensor(N_i, 4)],
        "labels": list[Tensor(N_i)],
        "texts": list[list[str]],
        "image_ids": list[int]
      }
    """
    images = torch.stack([b["image"] for b in batch], dim=0)

    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    texts = [b["texts"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "texts": texts,
        "image_ids": image_ids,
    }