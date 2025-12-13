"""
Zero-Shot Evaluation Script for YOLO-World - FIXED
Proper box denormalization and label positioning
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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.yolo_world import build_yolo_world
from utils.utils import load_config, setup_logger
from utils.device import get_available_device


class ZeroShotEvaluator:
    """Evaluator for zero-shot detection with custom vocabulary"""
    
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
        
        self.logger = setup_logger('logs', 'zero_shot_eval')
        
        # Build model
        self.logger.info("Building YOLO-World model...")
        try:
            self.model = build_yolo_world(config, self.device)
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Performance tracking
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
                    self.logger.info(f"✅ Loaded from epoch {checkpoint['epoch']}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.logger.info("✅ Checkpoint loaded successfully")
            
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
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 300
    ) -> Tuple[Dict, tuple, float]:
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
        
        # CRITICAL FIX: Properly denormalize boxes
        if len(pred['boxes']) > 0:
            max_val = pred['boxes'].max().item()
            
            # If max value is <= 1.5, boxes are normalized [0,1]
            if max_val <= 1.5:
                # Denormalize from [0,1] to model input size (640x640)
                img_size = self.config['data']['img_size']
                pred['boxes'][:, [0, 2]] *= img_size  # x coords
                pred['boxes'][:, [1, 3]] *= img_size  # y coords
            
            # Now scale from model input size to original image size
            img_size = self.config['data']['img_size']
            scale_x = original_size[0] / img_size  # original_width / 640
            scale_y = original_size[1] / img_size  # original_height / 640
            
            pred['boxes'][:, [0, 2]] *= scale_x
            pred['boxes'][:, [1, 3]] *= scale_y
            
            # Clamp to image boundaries
            pred['boxes'][:, [0, 2]] = pred['boxes'][:, [0, 2]].clamp(0, original_size[0])
            pred['boxes'][:, [1, 3]] = pred['boxes'][:, [1, 3]].clamp(0, original_size[1])
        
        self.total_inference_time += inference_time
        self.total_images += 1
        
        return pred, original_size, inference_time
    
    def visualize_predictions(
        self,
        image_path: str,
        predictions: Dict,
        category_names: List[str],
        save_path: str = None
    ):
        """Visualize predictions with labels ON TOP of boxes"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # Get predictions
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        # Handle no detections
        if len(boxes) == 0:
            ax.set_title("No detections", fontsize=14, pad=10)
            ax.axis('off')
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                self.logger.info(f"Saved: {save_path}")
            plt.close()
            return
        
        # Generate colors
        colors = plt.cm.hsv(np.linspace(0, 1, len(category_names)))
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Validation
            if w <= 0 or h <= 0:
                continue
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=3,
                edgecolor=colors[label],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label ON TOP of box
            label_text = f"{category_names[label]}: {score:.2f}"
            
            # Position label above box, but keep it in image bounds
            label_y = max(15, y1 - 5)  # At least 15 pixels from top
            
            ax.text(
                x1, label_y,
                label_text,
                bbox=dict(facecolor=colors[label], alpha=0.8, edgecolor='white', linewidth=1, pad=4),
                fontsize=11,
                color='white',
                weight='bold',
                verticalalignment='bottom'
            )
        
        ax.set_title(f"Detections: {len(boxes)} objects", fontsize=14, pad=10)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Saved: {save_path}")
        
        plt.close()
    
    def evaluate_images(
        self,
        image_paths: List[str],
        category_names: List[str],
        output_dir: str = 'outputs/zero_shot',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        save_viz: bool = True
    ):
        """Evaluate on multiple images"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Evaluating {len(image_paths)} images")
        self.logger.info(f"Categories: {category_names}")
        
        total_detections = 0
        
        for img_path in tqdm(image_paths, desc="Evaluating"):
            try:
                # Predict
                predictions, original_size, inference_time = self.predict(
                    str(img_path),
                    category_names=category_names,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                img_name = Path(img_path).stem
                num_detections = len(predictions['boxes'])
                total_detections += num_detections
                
                # Visualize
                if save_viz:
                    viz_path = os.path.join(output_dir, f"{img_name}_pred.jpg")
                    self.visualize_predictions(
                        str(img_path), predictions, category_names, viz_path
                    )
                
                # Log
                detected_cats = [category_names[l] for l in predictions['labels'].cpu().numpy()]
                self.logger.info(
                    f"{img_name}: {num_detections} detections in {inference_time:.3f}s | "
                    f"Categories: {detected_cats}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to process {img_path}: {e}")
                continue
        
        # Summary
        avg_time = self.total_inference_time / self.total_images if self.total_images > 0 else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"  Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total images: {len(image_paths)}")
        print(f"Total detections: {total_detections}")
        print(f"Avg inference time: {avg_time:.3f}s")
        print(f"FPS: {fps:.1f}")
        print(f"Results in: {output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Evaluation for YOLO-World')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--categories', type=str, nargs='+', required=True)
    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--iou_threshold', type=float, default=0.7)
    parser.add_argument('--output_dir', type=str, default='outputs/zero_shot')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create evaluator
    evaluator = ZeroShotEvaluator(config, args.checkpoint, args.device)
    
    # Get image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in exts:
            image_paths.extend(list(image_dir.glob(ext)))
    else:
        raise ValueError("Must provide either --image or --image_dir")
    
    print(f"\n📸 Found {len(image_paths)} images\n")
    
    # Evaluate
    evaluator.evaluate_images(
        image_paths,
        category_names=args.categories,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    print("✅ Evaluation complete!")


if __name__ == '__main__':
    main()