"""
Zero-Shot Evaluation Script for YOLO-World
Test the model with custom vocabulary not in COCO
"""

import os
import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.yolo_world import build_yolo_world
from utils.utils import load_config, setup_logger
from utils.device import get_available_device, print_device_info


class ZeroShotEvaluator:
    """Evaluator for zero-shot detection with custom vocabulary"""
    
    def __init__(self, config: dict, checkpoint_path: str, device: str = 'auto'):
        self.config = config
        
        # Setup device with auto-detection
        if device == 'auto':
            self.device, device_name = get_available_device('auto')
            print(f"\nAuto-detected device: {device_name}\n")
        else:
            self.device, device_name = get_available_device(device)
            print(f"\nUsing device: {device_name}\n")
        
        self.logger = setup_logger('logs', 'zero_shot_eval')
        
        # Build model
        self.logger.info("Building YOLO-World model...")
        self.model = build_yolo_world(config, self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.logger.info("Checkpoint loaded successfully")
    
    def preprocess_image(self, image_path: str, img_size: int = 640):
        """Preprocess image for inference"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (W, H)
        
        # Resize
        img = img.resize((img_size, img_size), Image.BILINEAR)
        
        # To tensor
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        
        return img_tensor.unsqueeze(0), original_size
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        category_names: list,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 300
    ):
        """
        Predict objects in image with custom vocabulary
        
        Args:
            image_path: Path to image
            category_names: List of category names to detect
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections
        
        Returns:
            predictions: Dictionary with detections
        """
        # Preprocess
        img_tensor, original_size = self.preprocess_image(
            image_path,
            self.config['data']['img_size']
        )
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        predictions = self.model.predict(
            img_tensor,
            category_names=category_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det
        )
        
        # Scale boxes back to original size
        img_size = self.config['data']['img_size']
        scale_x = original_size[0] / img_size
        scale_y = original_size[1] / img_size
        
        for pred in predictions:
            if len(pred['boxes']) > 0:
                pred['boxes'][:, [0, 2]] *= scale_x
                pred['boxes'][:, [1, 3]] *= scale_y
        
        return predictions[0], original_size
    
    def visualize_predictions(
        self,
        image_path: str,
        predictions: dict,
        category_names: list,
        save_path: str = None
    ):
        """Visualize predictions on image"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw boxes
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        colors = plt.cm.hsv(np.linspace(0, 1, len(category_names)))
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor=colors[label],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{category_names[label]}: {score:.2f}"
            ax.text(
                x1, y1 - 5,
                label_text,
                bbox=dict(facecolor=colors[label], alpha=0.5),
                fontsize=10,
                color='white'
            )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            self.logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def evaluate_images(
        self,
        image_paths: list,
        category_names: list,
        output_dir: str = 'outputs/zero_shot'
    ):
        """Evaluate on multiple images"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Evaluating {len(image_paths)} images")
        self.logger.info(f"Custom vocabulary: {category_names}")
        
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # Predict
            predictions, original_size = self.predict(
                img_path,
                category_names=category_names,
                conf_threshold=0.25
            )
            
            # Save visualization
            img_name = Path(img_path).stem
            save_path = os.path.join(output_dir, f"{img_name}_pred.jpg")
            
            self.visualize_predictions(
                img_path,
                predictions,
                category_names,
                save_path=save_path
            )
            
            # Log results
            num_detections = len(predictions['boxes'])
            self.logger.info(
                f"{img_name}: {num_detections} detections | "
                f"Categories: {[category_names[l] for l in predictions['labels'].cpu().numpy()]}"
            )


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Evaluation for YOLO-World')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to directory of images')
    parser.add_argument('--categories', type=str, nargs='+', required=True,
                        help='Custom category names to detect')
    parser.add_argument('--output_dir', type=str, default='outputs/zero_shot',
                        help='Output directory for visualizations')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, mps, or cpu')
    
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
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    else:
        raise ValueError("Must provide either --image or --image_dir")
    
    # Evaluate
    evaluator.evaluate_images(
        image_paths,
        category_names=args.categories,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()