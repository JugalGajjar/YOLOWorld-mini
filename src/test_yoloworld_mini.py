"""
Test script for YOLOWorldMini model
"""

import torch

from utils.config import Config
from modules.yolo_world_model import YOLOWorldMini


def main():
    cfg = Config()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = YOLOWorldMini(cfg).to(device)
    model.eval()

    # fake input
    x = torch.randn(1, 3, 640, 640, device=device)
    vocab = ["person", "bicycle", "cat", "dog"]

    with torch.no_grad():
        boxes, logits, region_embs = model(x, vocab_texts=vocab, device=device)

    print("boxes shape:", boxes.shape)  # (B, K, 4)
    print("logits shape:", logits.shape)  # (B, K, C_vocab)
    print("region_embs shape:", region_embs.shape)  # (B, K, D)


if __name__ == "__main__":
    main()