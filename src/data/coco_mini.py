"""
COCO-mini dataset class
"""

from typing import Dict, Any, List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms


class COCOMiniDataset(Dataset):
    """
    COCO-mini dataset:
      - uses a subset JSON (instances_train2017_mini.json)
      - returns image tensor (3, 640, 640), boxes in absolute coords,
        labels, and text category names.
    """

    def __init__(self, images_dir: Path, ann_file: Path, transforms_fn=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(ann_file))
        self.transforms_fn = transforms_fn

        self.img_ids = list(self.coco.imgs.keys())
        self.cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

        # default transform: resize to 640x640, ToTensor
        if self.transforms_fn is None:
            self.transforms_fn = transforms.Compose(
                [
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),   # [0,1], shape (3, H, W)
                ]
            )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # annotations
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

        # convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # apply image transform and scale boxes accordingly
        image_t = self.transforms_fn(image)  # (3, 640, 640)
        _, new_h, new_w = image_t.shape

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        if boxes.numel() > 0:
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

        sample: Dict[str, Any] = {
            "image": image_t,
            "boxes": boxes,
            "labels": labels,
            "texts": texts,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
            "new_size": (new_h, new_w),
        }

        return sample