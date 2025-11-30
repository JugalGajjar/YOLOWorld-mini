"""
Configuration for model, data, and training settings
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    coco_root: Path = Path("data/coco")
    coco_mini_annot: Path = Path("data/coco/annotations/instances_train2017_mini.json")
    coco_mini_images: Path = Path("data/coco/train2017")
    num_classes: int = 80  # COCO classes


@dataclass
class Config:
    data: DataConfig = DataConfig()