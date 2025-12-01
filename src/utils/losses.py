"""
Loss functions for YOLOWorld-mini model
"""

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F


def build_batch_vocab(batch_texts: List[List[str]]) -> Tuple[List[str], List[List[int]]]:
    """
    Given batch_texts, e.g.
      batch_texts = [["person", "dog"], ["person", "car", "bicycle"],]
    returns:
      batch_vocab: sorted list of unique category names in this batch
      per_image_maps: list of lists mapping per-image index -> batch_vocab index

    Example:
      batch_vocab = ["bicycle", "car", "dog", "person"]
      per_image_maps[0] = [3, 2]  # "person" -> 3, "dog" -> 2
      per_image_maps[1] = [3, 1, 0]
    """
    vocab_set = set()
    for texts in batch_texts:
        vocab_set.update(texts)
    batch_vocab = sorted(vocab_set)

    # map name -> index
    name_to_idx = {name: i for i, name in enumerate(batch_vocab)}

    per_image_maps: List[List[int]] = []
    for texts in batch_texts:
        per_image_maps.append([name_to_idx[t] for t in texts])

    return batch_vocab, per_image_maps


def _assign_cell_and_anchor(
    gt_box: torch.Tensor,
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    num_anchors: int,
) -> int:
    """
    Very simple YOLO-style assignment:
    - Compute center of GT box in image coords.
    - Map to feature map cell (iy, ix) at resolution H_feat x W_feat.
    - Use anchor 0 for now.
    Returns:
      region_idx in [0, H_feat * W_feat * num_anchors)
    """
    x1, y1, x2, y2 = gt_box.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    # normalize to [0, 1], then to grid
    gx = cx / img_w
    gy = cy / img_h

    ix = min(W_feat - 1, max(0, int(gx * W_feat)))
    iy = min(H_feat - 1, max(0, int(gy * H_feat)))

    cell_idx = iy * W_feat + ix
    anchor_idx = 0  # you can later experiment with smarter anchor choice
    region_idx = cell_idx * num_anchors + anchor_idx
    return region_idx


def yolo_world_classification_loss(
    boxes_pred: torch.Tensor,  # (B, K, 4), predicted boxes in xyxy (image coords)
    logits: torch.Tensor,  # (B, K, V), class logits for batch vocab
    batch_boxes: List[torch.Tensor],  # length B, each (N_i, 4) in xyxy (image coords)
    batch_texts: List[List[str]],  # length B, list of category names per GT box
    batch_vocab: List[str],  # length V, global vocab for this batch
    per_image_maps: List[List[int]],  # length B, indices mapping per-image text -> vocab idx
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    num_anchors: int = 3,
    lambda_box: float = 1.0,
) -> torch.Tensor:
    """
    Detection loss:
      - classification: BCE-with-logits on the vocab dimension for assigned region
      - box regression: Smooth L1 between predicted box and GT box at assigned region

    We keep the function name `yolo_world_classification_loss` so your train.py
    does not need a big refactor, but now it includes a box loss as well.
    """

    device = logits.device
    B, K, V = logits.shape

    # Basic checks
    assert V == len(batch_vocab), "Vocab size mismatch between logits and batch_vocab"

    total_cls_loss = torch.zeros((), device=device)
    total_box_loss = torch.zeros((), device=device)
    num_pos = 0

    for b in range(B):
        gt_boxes = batch_boxes[b]  # (N_i, 4)
        if gt_boxes.numel() == 0:
            continue

        gt_texts = batch_texts[b]  # list[str]
        gt_vocab_idxs = per_image_maps[b]  # list[int], same length as gt_texts

        for j, gt_box in enumerate(gt_boxes):
            v_idx = gt_vocab_idxs[j]  # vocab index for this GT box

            # assign one region index for this GT
            region_idx = _assign_cell_and_anchor(
                gt_box=gt_box,
                H_feat=H_feat,
                W_feat=W_feat,
                img_h=img_h,
                img_w=img_w,
                num_anchors=num_anchors,
            )

            if region_idx < 0 or region_idx >= K:
                # should not normally happen, but be safe
                continue

            # classification target: all zeros except 1 at v_idx
            target_cls = torch.zeros((V,), device=device)
            target_cls[v_idx] = 1.0

            pred_logits = logits[b, region_idx]  # (V,)
            cls_loss = F.binary_cross_entropy_with_logits(pred_logits, target_cls)

            # box regression target: smooth L1 between predicted box and GT box
            pred_box = boxes_pred[b, region_idx]  # (4,)
            box_loss = F.smooth_l1_loss(pred_box, gt_box.to(device), reduction="mean")

            total_cls_loss += cls_loss
            total_box_loss += box_loss
            num_pos += 1

    if num_pos == 0:
        # no objects in this batch: loss is 0, but you might also want to
        # add some negative-only classification loss later
        return total_cls_loss + lambda_box * total_box_loss

    total_cls_loss = total_cls_loss / num_pos
    total_box_loss = total_box_loss / num_pos

    total_loss = total_cls_loss + lambda_box * total_box_loss
    return total_loss