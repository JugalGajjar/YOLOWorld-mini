"""
Training script for YOLOWorld-mini model on COCO-Mini dataset with detection + region–text contrastive loss.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.utils as nn_utils

from utils.config import Config
from utils.collate import coco_collate
from utils.losses import build_batch_vocab, yolo_world_loss
from utils.logging_utils import get_logger
from data.coco_mini import COCOMiniDataset
from modules.yolo_world_model import YOLOWorldMini


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

    # Dataset & loader
    train_dataset = COCOMiniDataset(
        images_dir=cfg.data.coco_mini_images,
        ann_file=cfg.data.coco_mini_annot,
        img_size=cfg.data.img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=coco_collate,
    )

    # Model
    model = YOLOWorldMini(cfg).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    H_feat = cfg.data.img_size // 8
    W_feat = cfg.data.img_size // 8
    img_h = cfg.data.img_size
    img_w = cfg.data.img_size

    global_step = 0
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            images = batch["images"].to(device)  # (B, 3, img_size, img_size)
            batch_boxes = batch["boxes"]  # list[Tensor(N_i, 4)]
            batch_texts = batch["texts"]  # list[list[str]]

            # Build vocab for this batch
            vocab_texts, per_image_maps = build_batch_vocab(batch_texts)

            # Forward
            boxes_pred, logits, obj_logits, region_embs, text_embs = model(
                images,
                vocab_texts=vocab_texts,
                device=device,
            )

            total_loss, cls_loss, obj_loss, box_loss, contrast_loss = yolo_world_loss(
                boxes_pred=boxes_pred,
                logits=logits,
                obj_logits=obj_logits,
                region_embs=region_embs,
                text_embs=text_embs,
                batch_boxes=batch_boxes,
                batch_texts=batch_texts,
                batch_vocab=vocab_texts,
                per_image_maps=per_image_maps,
                H_feat=H_feat,
                W_feat=W_feat,
                img_h=img_h,
                img_w=img_w,
                lambda_obj=cfg.train.lambda_obj,
                lambda_box=cfg.train.lambda_box,
                lambda_contrast=cfg.train.lambda_contrast,
                contrast_temperature=cfg.train.contrast_temperature,
            )

            optimizer.zero_grad()
            total_loss.backward()
            if cfg.train.max_grad_norm is not None and cfg.train.max_grad_norm > 0:
                nn_utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()

            if global_step % cfg.train.log_every == 0:
                logger.info(
                    f"Epoch {epoch} step {global_step} "
                    f"loss={total_loss.item():.4f} "
                    f"cls={cls_loss.item():.4f} "
                    f"obj={obj_loss.item():.4f} "
                    f"box={box_loss.item():.4f} "
                    f"ctr={contrast_loss.item():.4f}"
                )

            global_step += 1

        # Save checkpoint every epoch
        out_dir = cfg.train.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"yoloworld_mini_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()