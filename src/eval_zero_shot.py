"""
Zero-shot evaluation script for YOLOWorld-mini model on COCO dataset.

Features:
- Samples up to 1000 images from the original COCO val2017 split
  (not the coco-mini subset).
- For each checkpoint in cfg.train.output_dir matching 'yoloworld_mini_epoch*.pt':
    * Computes a simple image-level top-1 accuracy:
        - Build vocab from ground-truth category names in the image.
        - Model predicts one category per image (over all regions and classes).
        - A prediction is counted correct if the predicted category is present
          in the ground-truth categories of that image.
    * Saves a few visualization images with top predicted box + label.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import ImageDraw

from utils.config import Config
from utils.collate import coco_collate
from utils.logging_utils import get_logger
from data.coco_mini import COCOMiniDataset
from modules.yolo_world_model import YOLOWorldMini


def tensor_to_pil(t: torch.Tensor) -> "Image.Image":
    """
    Convert a (3, H, W) tensor in [0,1] to a PIL image.
    """
    to_pil = transforms.ToPILImage()
    return to_pil(t.cpu())


def draw_single_detection(image_t: torch.Tensor, box: torch.Tensor, label: str,
                          score: float) -> "Image.Image":
    """
    Draw a single bounding box and label on the image.
    image_t: (3, H, W) in [0,1]
    box: (4,) xyxy in absolute coords (may be unordered, so we sanitize it)
    """
    img = tensor_to_pil(image_t)
    draw = ImageDraw.Draw(img)
    W, H = img.size  # note: PIL uses (width, height)

    x1, y1, x2, y2 = box.tolist()

    # 1) ensure x1 <= x2, y1 <= y2
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # 2) clamp to image bounds
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))

    # 3) if degenerate box, just return image without drawing
    if x2 <= x1 or y2 <= y1:
        return img

    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    text = f"{label} ({score:.2f})"
    draw.text((x1 + 4, y1 + 4), text, fill="red")

    return img


def evaluate_checkpoint(cfg: Config, ckpt_path: Path, dataset: Subset, device: torch.device,
                        logger, max_vis_images: int = 8) -> float:
    """
    Evaluate a single checkpoint on the given dataset subset.
    Returns:
      image-level top-1 accuracy.
    Also saves a few visualization images.
    """

    logger.info(f"Evaluating checkpoint: {ckpt_path.name}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=coco_collate,
    )

    model = YOLOWorldMini(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total = 0
    correct = 0

    vis_dir = cfg.train.output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)  # (B, 3, 640, 640)
            batch_texts = batch["texts"]  # list[list[str]]
            image_ids = batch["image_ids"]  # list[int]

            # Build vocab for this batch from GT category names
            all_names = sorted({t for texts in batch_texts for t in texts})
            if len(all_names) == 0:
                # skip images with no annotations
                continue

            boxes_pred, logits, region_embs = model(
                images, vocab_texts=all_names, device=device
            )

            B, K, C = logits.shape
            probs = torch.softmax(logits, dim=-1)

            # For each image:
            for b in range(B):
                gt_names = set(batch_texts[b])
                if len(gt_names) == 0:
                    continue

                # Find global top prediction over regions and classes
                probs_b = probs[b]  # (K, C)
                top_val, top_idx = probs_b.view(-1).max(dim=0)
                flat_idx = top_idx.item()
                region_idx = flat_idx // C
                class_idx = flat_idx % C

                pred_name = all_names[class_idx]
                score = top_val.item()

                total += 1
                if pred_name in gt_names:
                    correct += 1

                # Save a few visualizations
                if vis_count < max_vis_images:
                    box = boxes_pred[b, region_idx]  # (4,)
                    img_t = images[b].cpu()
                    img_vis = draw_single_detection(img_t, box, pred_name, score)

                    out_name = (
                        f"{ckpt_path.stem}_img{image_ids[b]}_pred_{pred_name.replace(' ', '_')}.jpg"
                    )
                    img_vis.save(vis_dir / out_name)
                    vis_count += 1

    acc = correct / total if total > 0 else 0.0
    logger.info(
        f"Checkpoint {ckpt_path.name}: accuracy={acc:.4f} "
        f"({correct}/{total} images with correct top-1 category)"
    )
    return acc


def main():
    cfg = Config()
    logger = get_logger()

    device = torch.device(
        cfg.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Use COCO val2017 split (not the mini subset)
    coco_root = cfg.data.coco_root
    val_images = coco_root / "val2017"
    val_annot = coco_root / "annotations" / "instances_val2017.json"

    if not val_images.exists():
        raise FileNotFoundError(f"COCO val images not found at: {val_images}")
    if not val_annot.exists():
        raise FileNotFoundError(f"COCO val annotations not found at: {val_annot}")

    # Build dataset on full COCO val
    full_val_dataset = COCOMiniDataset(
        images_dir=val_images,
        ann_file=val_annot,
    )

    # Randomly sample up to 1000 images
    num_samples = min(1000, len(full_val_dataset))
    indices = random.sample(range(len(full_val_dataset)), k=num_samples)
    subset = Subset(full_val_dataset, indices)
    logger.info(f"Evaluating on {num_samples} randomly sampled COCO val images.")

    # Find all checkpoints in output_dir
    ckpt_dir = cfg.train.output_dir
    ckpt_paths = sorted(ckpt_dir.glob("yoloworld_mini_epoch*.pt"))

    if not ckpt_paths:
        logger.error(f"No checkpoints found in {ckpt_dir}")
        return

    best_acc = -1.0
    best_ckpt = None

    for ckpt_path in ckpt_paths:
        acc = evaluate_checkpoint(cfg, ckpt_path, subset, device, logger)
        if acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt_path

    logger.info(
        f"Best checkpoint: {best_ckpt.name if best_ckpt is not None else 'None'} "
        f"with accuracy={best_acc:.4f}"
    )
    logger.info(
        f"Visualizations saved to: {cfg.train.output_dir / 'vis'}"
    )


if __name__ == "__main__":
    main()