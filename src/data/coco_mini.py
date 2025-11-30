"""
COCO-mini dataset class
"""

from typing import Dict, Any, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class COCOMiniDataset(Dataset):
    """
    COCO-mini dataset:
      - uses a subset JSON (instances_train2017_mini.json)
      - returns image, boxes, labels, and text category names
    """

    def __init__(self, images_dir: Path, ann_file: Path, transforms=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(ann_file))
        self.transforms = transforms

        self.img_ids = list(self.coco.imgs.keys())
        self.cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        texts: List[str] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            texts.append(self.cat_id_to_name[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        sample: Dict[str, Any] = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "texts": texts,
            "image_id": img_id,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample