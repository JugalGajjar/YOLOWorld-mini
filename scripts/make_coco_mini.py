"""
Making a mini COCO dataset from the full COCO dataset
"""

import json
from pathlib import Path

def make_coco_mini(
    ann_path: Path,
    out_path: Path,
    max_images: int = 10000,
):
    ann = json.loads(ann_path.read_text())
    images = ann["images"]
    annotations = ann["annotations"]
    categories = ann["categories"]

    # Take first max_images images
    mini_images = images[:max_images]
    mini_image_ids = {img["id"] for img in mini_images}

    mini_annotations = [
        a for a in annotations if a["image_id"] in mini_image_ids
    ]

    mini = {
        "info": ann.get("info", {}),
        "licenses": ann.get("licenses", []),
        "images": mini_images,
        "annotations": mini_annotations,
        "categories": categories,
    }

    out_path.write_text(json.dumps(mini))
    print(f"Saved mini COCO to {out_path}")
    print(f"Images: {len(mini_images)}, Annotations: {len(mini_annotations)}")


if __name__ == "__main__":
    root = Path("data/coco/annotations")
    ann_path = root / "instances_train2017.json"
    out_path = root / "instances_train2017_mini.json"

    make_coco_mini(ann_path, out_path, max_images=20000)