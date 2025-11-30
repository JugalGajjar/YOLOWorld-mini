"""
Test script for COCOMiniDataset
"""

from pathlib import Path

from data.coco_mini import COCOMiniDataset
from utils.config import Config


def main():
    cfg = Config()
    ds = COCOMiniDataset(
        images_dir=cfg.data.coco_mini_images,
        ann_file=cfg.data.coco_mini_annot,
    )

    print("Dataset length:", len(ds))
    sample = ds[0]
    print("Keys:", sample.keys())
    print("Image:", sample["image"])
    print("Boxes shape:", sample["boxes"].shape)
    print("Labels:", sample["labels"])
    print("Texts:", sample["texts"][:5])  # first few category names


if __name__ == "__main__":
    main()