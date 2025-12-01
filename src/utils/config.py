"""
Configuration for model, data, and training settings
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    backbone_variant: str = "models/yolov8s.pt"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    feat_channels = (144, 144, 144)  # P3, P4, P5 from YOLOv8s backbone
    embed_dim: int = 512  # CLIP text embedding dim
    num_anchors: int = 3


@dataclass
class DataConfig:
    coco_root: Path = Path("data/coco")
    coco_mini_annot: Path = Path("data/coco/annotations/instances_train2017_mini.json")
    coco_mini_images: Path = Path("data/coco/train2017")
    num_classes: int = 80  # COCO classes


@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-4
    num_workers: int = 4
    max_vocab_size: int = 80
    log_interval: int = 50
    save_interval: int = 5
    output_dir: Path = Path("outputs")


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()