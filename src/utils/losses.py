"""
Loss functions for YOLOWorld-mini model
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def build_batch_vocab(batch_texts: List[List[str]]) -> Tuple[List[str], List[dict]]:
    """
    Build a vocabulary list from batch texts.
    Returns:
      vocab: list[str]
      maps: list[dict[str, int]] one per image mapping category name -> vocab index
    """
    all_names = set()
    for texts in batch_texts:
        all_names.update(texts)

    vocab = sorted(all_names)
    name_to_idx = {name: i for i, name in enumerate(vocab)}

    per_image_maps = []
    for texts in batch_texts:
        m = {t: name_to_idx[t] for t in texts}
        per_image_maps.append(m)

    return vocab, per_image_maps


def assign_targets_to_cells(
    boxes: torch.Tensor,
    texts: List[str],
    text_to_vocab_idx: dict,
    H_feat: int,
    W_feat: int,
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For one image:
      - boxes: (N, 4) in absolute image coords
      - texts: list[str] category names per box
      - text_to_vocab_idx: mapping str -> vocab index

    Returns:
      pos_indices: (N_pos,) indices in [0, H_feat*W_feat) for anchor 0
      pos_labels: (N_pos,) vocab indices
    """
    if boxes.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=boxes.device),
            torch.zeros((0,), dtype=torch.long, device=boxes.device),
        )

    # compute box centers in image coords
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # map centers to feature map indices
    stride_x = img_w / W_feat
    stride_y = img_h / H_feat

    gx = torch.clamp((cx / stride_x).long(), 0, W_feat - 1)
    gy = torch.clamp((cy / stride_y).long(), 0, H_feat - 1)

    cell_idx = gy * W_feat + gx   # (N,)

    # vocab indices
    labels = [text_to_vocab_idx[t] for t in texts]
    labels = torch.tensor(labels, dtype=torch.long, device=boxes.device)

    return cell_idx, labels


def yolo_world_classification_loss(
    boxes_pred: torch.Tensor,    # (B, K, 4) not used now, but kept for extensibility
    logits: torch.Tensor,        # (B, K, C_vocab)
    batch_boxes: List[torch.Tensor],
    batch_texts: List[List[str]],
    batch_vocab: List[str],
    per_image_maps: List[dict],
    H_feat: int,
    W_feat: int,
    img_h: int = 640,
    img_w: int = 640,
) -> torch.Tensor:
    """
    Compute simple classification loss:
      - For each GT box, find its cell index (anchor 0)
      - Use CE loss on logits for that cell vs its vocab index.

    boxes_pred is currently unused but passed for future extension.
    """
    device = logits.device
    B, K, C_vocab = logits.shape
    assert C_vocab == len(batch_vocab)

    logits_flat = logits.view(B * K, C_vocab)

    pos_indices_all = []
    pos_labels_all = []

    for b in range(B):
        boxes = batch_boxes[b].to(device)
        texts = batch_texts[b]
        text_to_vocab_idx = per_image_maps[b]

        pos_cells, pos_labels = assign_targets_to_cells(
            boxes,
            texts,
            text_to_vocab_idx,
            H_feat,
            W_feat,
            img_h,
            img_w,
        )

        if pos_cells.numel() == 0:
            continue

        # anchor 0 only: index in [0, H*W) → global index shift
        base = b * K
        # K = num_anchors * H * W, but we only use anchor 0 → offset = cell_idx
        pos_indices_all.append(base + pos_cells)
        pos_labels_all.append(pos_labels)

    if not pos_indices_all:
        # no objects in this batch
        return torch.tensor(0.0, device=device, requires_grad=True)

    pos_indices = torch.cat(pos_indices_all, dim=0)
    pos_labels = torch.cat(pos_labels_all, dim=0)

    pos_logits = logits_flat[pos_indices]  # (N_pos, C_vocab)

    loss = F.cross_entropy(pos_logits, pos_labels)
    return loss