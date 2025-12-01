"""
Training script for YOLOWorld-mini model on COCO-Mini dataset
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.config import Config
from utils.collate import coco_collate
from utils.losses import build_batch_vocab, yolo_world_classification_loss
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

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)

    # assuming P3 has spatial size 80x80 for 640x640 input
    H_feat, W_feat = 80, 80

    global_step = 0
    for epoch in range(cfg.train.epochs):
        for batch in train_loader:
            images = batch["images"].to(device)  # (B, 3, 640, 640)
            batch_boxes = batch["boxes"]  # list[Tensor(N_i, 4)]
            batch_texts = batch["texts"]  # list[list[str]]

            # build vocab for this batch
            vocab_texts, per_image_maps = build_batch_vocab(batch_texts)

            # forward
            boxes_pred, logits, region_embs = model(images, vocab_texts=vocab_texts, device=device)

            # classification-only loss
            loss = yolo_world_classification_loss(
                boxes_pred=boxes_pred,
                logits=logits,
                batch_boxes=batch_boxes,
                batch_texts=batch_texts,
                batch_vocab=vocab_texts,
                per_image_maps=per_image_maps,
                H_feat=H_feat,
                W_feat=W_feat,
                img_h=640,
                img_w=640,
                num_anchors=cfg.model.num_anchors,
                lambda_box=1.0,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % cfg.train.log_interval == 0:
                logger.info(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

            global_step += 1

        # save checkpoint every epoch
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