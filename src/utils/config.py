"""
Configuration for model, data, training and evaluation settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    # We keep the name for clarity but the current backbone is a custom CNN.
    backbone_name: str = "mini-yolov8"
    # Channel dimension of the backbone's feature maps (P3, P4, P5).
    img_feat_dim: int = 256
    # Dimension of text / region embeddings.
    text_dim: int = 512
    # CLIP settings for text encoder (open_clip).
    clip_model_name: str = "ViT-B-32"
    clip_pretrained_tag: str = "openai"


@dataclass
class DataConfig:
    # Root of COCO (expects train2017 / val2017 / annotations).
    coco_root: Path = Path("data/coco")
    coco_mini_images: Path = Path("data/coco/train2017")
    coco_mini_annot: Path = Path("data/coco/annotations/instances_train2017_mini.json")
    # Square training size (e.g. 640).
    img_size: int = 640
    num_classes: int = 80  # COCO classes


@dataclass
class TrainConfig:
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    max_grad_norm: float = 1.0
    log_every: int = 50
    output_dir: Path = Path("outputs")

    # Detection loss weights.
    lambda_obj: float = 1.0
    lambda_box: float = 5.0

    # Region–text contrastive loss (InfoNCE).
    lambda_contrast: float = 1.0
    contrast_temperature: float = 0.07


@dataclass
class EvalConfig:
    # Threshold on per-image class score for “present”.
    label_thresh: float = 0.5
    # How many top cells to visualize per image.
    topk_vis: int = 20
    # How many images to evaluate on from COCO val.
    num_eval_images: int = 500
    # How many images to save visualizations for (same ids for all ckpts).
    num_vis_images: int = 10


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()