"""
Loss functions and helper utilities for YOLOWorld-mini.

This module provides:
- build_batch_vocab: construct a shared vocabulary for a batch of images
- assign_targets_to_cells: map GT boxes + labels to feature-map cells
- build_objectness_targets: generate objectness targets for BCE
- bbox_giou_aligned: GIoU for aligned box pairs
- region_text_contrastive_loss: InfoNCE region–text alignment
- yolo_world_loss: classification + objectness + GIoU + contrastive loss
"""

from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F


# Vocabulary utilities
def build_batch_vocab(batch_texts: List[List[str]]) -> Tuple[List[str], List[Dict[str, int]]]:
    """
    Build a shared vocabulary over all images in the batch, and a mapping
    from per-image category names to indices in that shared vocabulary.

    Args:
        batch_texts: list over images, each is a list of category names (strings)

    Returns:
        vocab: list of unique category names in the batch
        per_image_maps: list of dicts, one per image:
            name -> index in vocab
    """
    all_names = set()
    for texts in batch_texts:
        all_names.update(texts)
    vocab = sorted(all_names)

    per_image_maps: List[Dict[str, int]] = []
    name_to_idx = {name: i for i, name in enumerate(vocab)}

    for texts in batch_texts:
        mapping = {}
        for t in texts:
            if t in name_to_idx:
                mapping[t] = name_to_idx[t]
        per_image_maps.append(mapping)

    return vocab, per_image_maps


# Target assignment (grid-based P3)
def assign_targets_to_cells(
    boxes: torch.Tensor,
    texts: List[str],
    text_to_vocab_idx: Dict[str, int],
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    return_boxes: bool = False,
):
    """
    Assign each GT box to a single feature map cell on P3, using box center.

    Args:
        boxes: (N, 4) tensor in absolute pixel coords [x1, y1, x2, y2] on a resized
               image of size (img_w, img_h). Here typically 640 x 640.
        texts: list of length N, category names for each box.
        text_to_vocab_idx: mapping from category name to index in the shared vocab.
        H_feat, W_feat: spatial resolution of the feature map (e.g., 80 x 80).
        img_h, img_w: image size in pixels.
        return_boxes: if True, also return normalized GT boxes per positive cell.

    Returns:
        If return_boxes is False:
            pos_cells: 1D tensor of cell indices in [0, H_feat*W_feat)
            pos_labels: 1D tensor of vocab indices corresponding to those cells
        If return_boxes is True:
            pos_cells, pos_labels, pos_boxes_norm
            where pos_boxes_norm is (N_pos, 4) normalized xyxy in [0,1].
    """
    device = boxes.device
    if boxes.numel() == 0:
        empty_cells = torch.empty(0, dtype=torch.long, device=device)
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        if return_boxes:
            empty_boxes = torch.empty(0, 4, dtype=torch.float32, device=device)
            return empty_cells, empty_labels, empty_boxes
        else:
            return empty_cells, empty_labels

    # Compute box centers
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    # Map centers to [0, W_feat-1] x [0, H_feat-1]
    cell_x = torch.clamp((cx / img_w * W_feat).long(), 0, W_feat - 1)
    cell_y = torch.clamp((cy / img_h * H_feat).long(), 0, H_feat - 1)

    # Flatten grid to single index
    cell_idx = cell_y * W_feat + cell_x  # (N,)

    pos_cells: List[int] = []
    pos_labels: List[int] = []
    pos_boxes_norm: List[torch.Tensor] = []

    for i, name in enumerate(texts):
        if name not in text_to_vocab_idx:
            continue
        v_idx = text_to_vocab_idx[name]
        pos_cells.append(int(cell_idx[i].item()))
        pos_labels.append(int(v_idx))

        if return_boxes:
            # Normalize GT box to [0,1] xyxy
            bx1, by1, bx2, by2 = boxes[i]
            pos_boxes_norm.append(
                torch.tensor(
                    [
                        bx1 / img_w,
                        by1 / img_h,
                        bx2 / img_w,
                        by2 / img_h,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )

    if not pos_cells:
        empty_cells = torch.empty(0, dtype=torch.long, device=device)
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        if return_boxes:
            empty_boxes = torch.empty(0, 4, dtype=torch.float32, device=device)
            return empty_cells, empty_labels, empty_boxes
        else:
            return empty_cells, empty_labels

    pos_cells_t = torch.tensor(pos_cells, dtype=torch.long, device=device)
    pos_labels_t = torch.tensor(pos_labels, dtype=torch.long, device=device)

    if return_boxes:
        pos_boxes_t = torch.stack(pos_boxes_norm, dim=0)  # (N_pos, 4)
        return pos_cells_t, pos_labels_t, pos_boxes_t
    else:
        return pos_cells_t, pos_labels_t


def build_objectness_targets(
    batch_boxes: List[torch.Tensor],
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    K: int,
) -> torch.Tensor:
    """
    Build objectness targets for each image:

    - target = 1 at GT cells (where boxes are assigned)
    - target = 0 elsewhere

    We reuse the same center-based assignment logic as assign_targets_to_cells,
    but ignore category labels.

    Args:
        batch_boxes: list of length B, each is (N_i, 4) in pixel coords.
        H_feat, W_feat: feature map resolution.
        img_h, img_w: image size in pixels.
        K: feature map size (H_feat * W_feat).

    Returns:
        obj_targets: (B, K) tensor in {0,1}.
    """
    B = len(batch_boxes)
    obj_targets = torch.zeros((B, K), dtype=torch.float32)

    for b in range(B):
        boxes = batch_boxes[b]
        if boxes.numel() == 0:
            continue

        dummy_texts = ["obj"] * boxes.shape[0]
        dummy_map = {"obj": 0}

        pos_cells, _ = assign_targets_to_cells(
            boxes,
            dummy_texts,
            dummy_map,
            H_feat,
            W_feat,
            img_h,
            img_w,
            return_boxes=False,
        )
        if pos_cells.numel() > 0:
            obj_targets[b, pos_cells] = 1.0

    return obj_targets


# Box geometry: GIoU
def bbox_giou_aligned(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) for aligned boxes: pred[i] with target[i].
    Both in normalized xyxy, same coordinate system [0,1].

    Returns:
        giou: (N,) GIoU for each pair, in [-1, 1].
    """
    # Intersection
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas
    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_tgt = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

    union = area_pred + area_tgt - inter + 1e-6
    iou = inter / union

    # Smallest enclosing box
    xc1 = torch.min(pred[:, 0], target[:, 0])
    yc1 = torch.min(pred[:, 1], target[:, 1])
    xc2 = torch.max(pred[:, 2], target[:, 2])
    yc2 = torch.max(pred[:, 3], target[:, 3])

    cw = (xc2 - xc1).clamp(min=0)
    ch = (yc2 - yc1).clamp(min=0)
    area_c = cw * ch + 1e-6

    giou = iou - (area_c - union) / area_c
    return giou


# Region–text contrastive loss
def region_text_contrastive_loss(
    region_embs: torch.Tensor,
    text_embs: torch.Tensor,
    batch_boxes: List[torch.Tensor],
    batch_texts: List[List[str]],
    per_image_maps: List[Dict[str, int]],
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE region–text contrastive loss.

    For each image, we:
      - Assign GT boxes to grid cells (pos_cells, pos_labels)
      - Take region_embs[b, pos_cells] as image features
      - Take text_embs[pos_labels] as text features
      - Apply CLIP-style symmetric InfoNCE.

    Args:
        region_embs: (B, K, D) normalized region embeddings.
        text_embs: (C_vocab, D) normalized text embeddings.
    """
    device = region_embs.device
    B, K, D = region_embs.shape

    losses = []

    for b in range(B):
        boxes = batch_boxes[b].to(device)
        texts = batch_texts[b]
        if boxes.numel() == 0:
            continue

        text_to_vocab = per_image_maps[b]

        pos_cells, pos_labels = assign_targets_to_cells(
            boxes,
            texts,
            text_to_vocab,
            H_feat,
            W_feat,
            img_h,
            img_w,
            return_boxes=False,
        )

        if pos_cells.numel() == 0:
            continue

        img_z = region_embs[b, pos_cells]  # (N_pos, D)
        txt_z = text_embs[pos_labels]  # (N_pos, D)

        img_z = F.normalize(img_z, dim=-1)
        txt_z = F.normalize(txt_z, dim=-1)

        logits = img_z @ txt_z.t() / temperature  # (N_pos, N_pos)
        targets = torch.arange(logits.size(0), device=device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        losses.append(0.5 * (loss_i2t + loss_t2i))

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


# Full detection loss
def yolo_world_loss(
    boxes_pred: torch.Tensor,
    logits: torch.Tensor,
    obj_logits: torch.Tensor,
    region_embs: torch.Tensor,
    text_embs: torch.Tensor,
    batch_boxes: List[torch.Tensor],
    batch_texts: List[List[str]],
    batch_vocab: List[str],  # kept for API compatibility (not used directly)
    per_image_maps: List[Dict[str, int]],
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
    lambda_obj: float = 1.0,
    lambda_box: float = 5.0,
    lambda_contrast: float = 1.0,
    contrast_temperature: float = 0.07,
):
    """
    Combined loss for YOLOWorld-mini:

    - Classification loss (cross-entropy) on GT cells only.
    - Objectness loss (BCE with logits) on all cells (1 at GT, 0 elsewhere).
    - Box regression loss based on GIoU on GT cells.
    - Region–text contrastive InfoNCE loss on GT-positive pairs.

    Args:
        boxes_pred: (B, K, 4) normalized boxes in [0,1] (xyxy).
        logits: (B, K, C_vocab) classification logits (CLIP scores).
        obj_logits: (B, K) objectness logits.
        region_embs: (B, K, D) normalized.
        text_embs: (C_vocab, D) normalized.
    """
    device = logits.device
    B, K, C_vocab = logits.shape

    cls_losses = []
    box_losses = []

    # Per-image classification + box regression on positive cells
    for b in range(B):
        boxes = batch_boxes[b].to(device)
        texts = batch_texts[b]

        if boxes.numel() == 0:
            continue

        text_to_vocab = per_image_maps[b]

        pos_cells, pos_labels, pos_boxes_norm = assign_targets_to_cells(
            boxes,
            texts,
            text_to_vocab,
            H_feat,
            W_feat,
            img_h,
            img_w,
            return_boxes=True,
        )

        if pos_cells.numel() == 0:
            continue

        # Classification on positive cells
        logits_b = logits[b]  # (K, C_vocab)
        pos_logits = logits_b[pos_cells]  # (N_pos, C_vocab)
        cls_loss_b = F.cross_entropy(pos_logits, pos_labels.to(device))
        cls_losses.append(cls_loss_b)

        # Box regression on positive cells (GIoU loss)
        pred_boxes_b = boxes_pred[b]  # (K, 4), normalized [0,1]
        pred_pos_boxes = pred_boxes_b[pos_cells]  # (N_pos, 4)
        pred_pos_boxes = pred_pos_boxes.clamp(0.0, 1.0)

        gt_pos_boxes = pos_boxes_norm.to(device)  # (N_pos, 4), normalized [0,1]

        giou = bbox_giou_aligned(pred_pos_boxes, gt_pos_boxes)  # (N_pos,)
        box_loss_b = (1.0 - giou).mean()
        box_losses.append(box_loss_b)

    cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
    box_loss = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0, device=device)

    # Objectness loss on all cells
    K_feat = logits.shape[1]  # should equal H_feat * W_feat
    obj_targets = build_objectness_targets(
        batch_boxes=batch_boxes,
        H_feat=H_feat,
        W_feat=W_feat,
        img_h=img_h,
        img_w=img_w,
        K=K_feat,
    ).to(device)  # (B, K)

    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits.view(B, K_feat),
        obj_targets,
    )

    # Region–text contrastive loss
    contrast_loss = region_text_contrastive_loss(
        region_embs=region_embs,
        text_embs=text_embs,
        batch_boxes=batch_boxes,
        batch_texts=batch_texts,
        per_image_maps=per_image_maps,
        H_feat=H_feat,
        W_feat=W_feat,
        img_h=img_h,
        img_w=img_w,
        temperature=contrast_temperature,
    )

    total_loss = cls_loss + lambda_obj * obj_loss + lambda_box * box_loss + lambda_contrast * contrast_loss
    return total_loss, cls_loss, obj_loss, box_loss, contrast_loss