# YOLOWorld-Mini

A small-scale, educational re-implementation of **YOLO-World: Real-Time Open-Vocabulary Object Detection (CVPR 2024)**, built for CSCI 6527 (Computer Vision).

The goal is not to fully reproduce the original training regime, but to implement the core ideas on a manageable subset of COCO:

- Text-conditioned multi-scale feature fusion (simplified RepVL-PAN)
- Region–text contrastive classification head
- Online training vocabulary and offline inference vocabulary
- Basic zero-shot open-vocabulary behavior

## Project Goals

1. Implement a YOLOv8-S–style backbone as the base detector.
2. Integrate a CLIP text encoder for generating text embeddings.
3. Add a simplified RepVL-PAN neck and a region–text contrastive head.
4. Train on a COCO-mini subset and compare against a YOLOv8-S baseline.
5. Demonstrate zero-shot detection with custom text prompts.

## Environment

Recommended:
- Python 3.10+
- PyTorch + CUDA (T4 via Colab / 2xT4 via Kaggle)
- open_clip_torch for CLIP text encoder
- ultralytics for a YOLOv8 backbone reference

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reference

```
Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. (2024). Yolo-world: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16901-16911).
```

Open-access paper: [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf)<br>
Official repo: [YOLO-World](https://github.com/AILab-CVC/YOLO-World)