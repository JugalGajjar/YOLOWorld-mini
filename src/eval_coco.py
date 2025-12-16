"""
COCO Evaluation Script for YOLO-World
Proper box denormalization + Category ID mapping
"""

import os
import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import List, Dict, Tuple

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.yolo_world import build_yolo_world
from utils.utils import load_config, setup_logger
from utils.device import get_available_device


# COCO category IDs (80 classes, but IDs are not sequential)
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def get_coco_category_id(label_index: int) -> int:
    """Convert model label index (0-79) to COCO category ID"""
    if 0 <= label_index < len(COCO_CATEGORY_IDS):
        return COCO_CATEGORY_IDS[label_index]
    return label_index + 1  # Fallback


class COCOEvaluator:
    """COCO-style evaluation with proper metrics"""
    
    def __init__(self, config: dict, checkpoint_path: str, device: str = 'auto'):
        self.config = config
        
        # Setup device
        if device == 'auto':
            self.device, device_name = get_available_device('auto')
            print(f"\n{'='*60}")
            print(f"  Auto-detected device: {device_name}")
            print(f"{'='*60}\n")
        else:
            self.device, device_name = get_available_device(device)
            print(f"\n{'='*60}")
            print(f"  Using device: {device_name}")
            print(f"{'='*60}\n")
        
        self.logger = setup_logger('logs', 'coco_eval')
        
        # Build model
        self.logger.info("Building YOLO-World model...")
        try:
            self.model = build_yolo_world(config, self.device)
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise
        
        # Load checkpoint
        self.checkpoint_name = checkpoint_path.split('/')[-1].split('.')[0]
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Statistics
        self.total_inference_time = 0
        self.total_images = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    self.logger.info(f"Loaded from epoch {checkpoint['epoch']}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.logger.info("Checkpoint loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def preprocess_image(self, image_path: str, img_size: int = 640) -> Tuple[torch.Tensor, tuple]:
        """Preprocess image for inference"""
        try:
            img = Image.open(image_path).convert('RGB')
            original_size = img.size  # (W, H)
            
            # Resize
            img = img.resize((img_size, img_size), Image.BILINEAR)
            
            # To tensor
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
            
            return img_tensor.unsqueeze(0), original_size
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess {image_path}: {e}")
            raise
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        category_names: List[str],
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_det: int = 300
    ) -> Tuple[Dict, float]:
        """Predict objects in image - FIXED denormalization"""
        # Preprocess
        img_tensor, original_size = self.preprocess_image(
            image_path,
            self.config['data']['img_size']
        )
        img_tensor = img_tensor.to(self.device)
        
        # Predict with timing
        start_time = time.time()
        
        predictions = self.model.predict(
            img_tensor,
            category_names=category_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det
        )
        
        inference_time = time.time() - start_time
        
        pred = predictions[0]
        
        if len(pred['boxes']) > 0:
            max_val = pred['boxes'].max().item()
            
            # If max value is <= 1.5, boxes are normalized [0,1]
            if max_val <= 1.5:
                # Denormalize from [0,1] to model input size (640x640)
                img_size = self.config['data']['img_size']
                pred['boxes'][:, [0, 2]] *= img_size  # x coords
                pred['boxes'][:, [1, 3]] *= img_size  # y coords
            
            # Scale from model input size to original image size
            img_size = self.config['data']['img_size']
            scale_x = original_size[0] / img_size
            scale_y = original_size[1] / img_size
            
            pred['boxes'][:, [0, 2]] *= scale_x
            pred['boxes'][:, [1, 3]] *= scale_y
            
            # Clamp to image boundaries
            pred['boxes'][:, [0, 2]] = pred['boxes'][:, [0, 2]].clamp(0, original_size[0])
            pred['boxes'][:, [1, 3]] = pred['boxes'][:, [1, 3]].clamp(0, original_size[1])
        
        self.total_inference_time += inference_time
        self.total_images += 1
        
        return pred, inference_time
    
    def compute_per_category_metrics(self, coco_eval: COCOeval, coco_gt: COCO) -> Dict:
        """
        Compute per-category AP metrics
        
        Returns:
            Dict mapping category names to their metrics
        """
        # Get precision array
        precision = coco_eval.eval['precision']
        
        # Category IDs in COCO evaluation
        cat_ids = coco_eval.params.catIds
        
        per_category_metrics = {}
        
        for idx, cat_id in enumerate(cat_ids):
            # Get category info - CHECK IF EXISTS
            cat_list = coco_gt.loadCats(cat_id)
            if not cat_list:
                # Category doesn't exist, skip
                continue
            
            cat_name = cat_list[0]['name']
            
            # AP@0.5
            ap_50 = precision[0, :, idx, 0, 2].mean()
            
            # AP@0.5:0.95
            ap_50_95 = precision[:, :, idx, 0, 2].mean()
            
            # AP@0.75
            ap_75 = precision[5, :, idx, 0, 2].mean()
            
            # AP by size
            ap_small = precision[:, :, idx, 0, 2].mean()
            ap_medium = precision[:, :, idx, 1, 2].mean()
            ap_large = precision[:, :, idx, 2, 2].mean()
            
            # Handle NaN
            ap_50 = float(ap_50) if not np.isnan(ap_50) else 0.0
            ap_50_95 = float(ap_50_95) if not np.isnan(ap_50_95) else 0.0
            ap_75 = float(ap_75) if not np.isnan(ap_75) else 0.0
            ap_small = float(ap_small) if not np.isnan(ap_small) else 0.0
            ap_medium = float(ap_medium) if not np.isnan(ap_medium) else 0.0
            ap_large = float(ap_large) if not np.isnan(ap_large) else 0.0
            
            per_category_metrics[cat_name] = {
                'category_id': int(cat_id),
                'mAP_0.5': ap_50,
                'mAP_0.5:0.95': ap_50_95,
                'mAP_0.75': ap_75,
                'mAP_small': ap_small,
                'mAP_medium': ap_medium,
                'mAP_large': ap_large
            }
        
        return per_category_metrics
    
    def evaluate_coco(
        self,
        ann_file: str,
        img_dir: str,
        category_names: List[str],
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        save_results: bool = True,
        output_dir: str = 'outputs/coco_eval'
    ):
        """Evaluate on COCO dataset with official metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load COCO annotations
        self.logger.info(f"Loading COCO annotations from {ann_file}")
        coco_gt = COCO(ann_file)
        
        img_ids = sorted(coco_gt.getImgIds())
        self.logger.info(f"Evaluating on {len(img_ids)} images")
        
        # Collect predictions in COCO format
        coco_results = []
        
        for img_id in tqdm(img_ids, desc="Evaluating"):
            try:
                # Get image info
                img_info = coco_gt.loadImgs(img_id)[0]
                img_path = os.path.join(img_dir, img_info['file_name'])
                
                if not os.path.exists(img_path):
                    self.logger.warning(f"Image not found: {img_path}")
                    continue
                
                # Predict
                predictions, inference_time = self.predict(
                    img_path,
                    category_names=category_names,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Convert to COCO format
                boxes = predictions['boxes'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Validate box
                    if w <= 0 or h <= 0:
                        continue
                    
                    coco_results.append({
                        'image_id': int(img_id),
                        'category_id': get_coco_category_id(int(label)),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
                
            except Exception as e:
                self.logger.error(f"Failed to process image {img_id}: {e}")
                continue
        
        # Save results
        if save_results:
            results_file = os.path.join(output_dir, 'coco_results.json')
            with open(results_file, 'w') as f:
                json.dump(coco_results, f)
            self.logger.info(f"Saved results to {results_file}")
        
        # Run COCO evaluation
        if len(coco_results) == 0:
            self.logger.error("No predictions to evaluate!")
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Running COCO evaluation...")
        self.logger.info("="*60)
        
        # Load results
        coco_dt = coco_gt.loadRes(coco_results)
        
        # Create evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Compute per-category metrics
        self.logger.info("\n" + "="*60)
        self.logger.info("Computing per-category metrics...")
        self.logger.info("="*60)
        
        per_category_metrics = self.compute_per_category_metrics(coco_eval, coco_gt)
        
        # Print per-category summary
        sorted_categories = sorted(
            per_category_metrics.items(),
            key=lambda x: x[1]['mAP_0.5'],
            reverse=True
        )
        
        print(f"\n{'='*80}")
        print(f"  Top 10 Categories by mAP@0.5")
        print(f"{'='*80}")
        print(f"{'Category':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'mAP@0.75':<12}")
        print(f"{'-'*80}")
        for cat_name, metrics in sorted_categories[:10]:
            print(f"{cat_name:<20} {metrics['mAP_0.5']:<12.3f} {metrics['mAP_0.5:0.95']:<15.3f} {metrics['mAP_0.75']:<12.3f}")
        
        print(f"\n{'='*80}")
        print(f"  Bottom 10 Categories by mAP@0.5")
        print(f"{'='*80}")
        print(f"{'Category':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'mAP@0.75':<12}")
        print(f"{'-'*80}")
        for cat_name, metrics in sorted_categories[-10:]:
            print(f"{cat_name:<20} {metrics['mAP_0.5']:<12.3f} {metrics['mAP_0.5:0.95']:<15.3f} {metrics['mAP_0.75']:<12.3f}")
        print()
        
        # Print summary
        avg_time = self.total_inference_time / self.total_images if self.total_images > 0 else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"  Performance Summary")
        print(f"{'='*60}")
        print(f"Total images: {self.total_images}")
        print(f"Total detections: {len(coco_results)}")
        print(f"Avg inference time: {avg_time:.3f}s")
        print(f"FPS: {fps:.1f}")
        print(f"{'='*60}\n")
        
        # Save metrics
        metrics = {
            'overall': {
                'mAP_0.5:0.95': float(coco_eval.stats[0]),
                'mAP_0.5': float(coco_eval.stats[1]),
                'mAP_0.75': float(coco_eval.stats[2]),
                'mAP_small': float(coco_eval.stats[3]),
                'mAP_medium': float(coco_eval.stats[4]),
                'mAP_large': float(coco_eval.stats[5]),
                'AR_1': float(coco_eval.stats[6]),
                'AR_10': float(coco_eval.stats[7]),
                'AR_100': float(coco_eval.stats[8]),
                'AR_small': float(coco_eval.stats[9]),
                'AR_medium': float(coco_eval.stats[10]),
                'AR_large': float(coco_eval.stats[11]),
                'inference_time': avg_time,
                'fps': fps
            },
            'per_category': per_category_metrics
        }
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f'metrics_{self.checkpoint_name}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        
        # Save CSV
        csv_file = os.path.join(output_dir, f'per_category_{self.checkpoint_name}.csv')
        with open(csv_file, 'w') as f:
            f.write("Category,Category_ID,mAP@0.5,mAP@0.5:0.95,mAP@0.75,mAP_small,mAP_medium,mAP_large\n")
            for cat_name in sorted(per_category_metrics.keys()):
                m = per_category_metrics[cat_name]
                f.write(f"{cat_name},{m['category_id']},{m['mAP_0.5']:.4f},{m['mAP_0.5:0.95']:.4f},"
                       f"{m['mAP_0.75']:.4f},{m['mAP_small']:.4f},{m['mAP_medium']:.4f},{m['mAP_large']:.4f}\n")
        
        self.logger.info(f"Per-category CSV saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='COCO Evaluation for YOLO-World')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--categories_file', type=str, default=None)
    parser.add_argument('--conf_threshold', type=float, default=0.01)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='outputs/coco_eval')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Load category names
    if args.categories_file and os.path.exists(args.categories_file):
        with open(args.categories_file) as f:
            category_names = [line.strip() for line in f if line.strip()]
    else:
        # COCO categories (80 classes)
        category_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    print(f"\nCOCO Evaluation")
    print(f"Categories: {len(category_names)}")
    print(f"Annotation file: {args.ann_file}")
    print(f"Images directory: {args.img_dir}\n")
    
    evaluator = COCOEvaluator(config, args.checkpoint, args.device)
    
    evaluator.evaluate_coco(
        ann_file=args.ann_file,
        img_dir=args.img_dir,
        category_names=category_names,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete! Results in: {args.output_dir}\n")


if __name__ == '__main__':
    main()