"""
Video Demo Script for YOLO-World
Proper box denormalization and label display
"""

import os
import sys
import argparse
from pathlib import Path
import time
from typing import List

import torch
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.yolo_world import build_yolo_world
from utils.utils import load_config, setup_logger
from utils.device import get_available_device


class VideoDemo:
    """Real-time video demo for YOLO-World"""
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
        
        self.logger = setup_logger('logs', 'video_demo')
        
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
        self.fps_history = []
    
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
    
    def preprocess_frame(self, frame: np.ndarray, img_size: int = 640):
        """Preprocess video frame for inference"""
        original_h, original_w = frame.shape[:2]
        
        # Resize
        resized = cv2.resize(frame, (img_size, img_size))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # To tensor
        img_tensor = torch.from_numpy(rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) # (3, H, W)
        
        return img_tensor.unsqueeze(0), (original_w, original_h)
    
    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray, category_names: List[str], conf_threshold: float = 0.25,
                      iou_threshold: float = 0.7):
        """Predict objects in frame"""
        # Preprocess
        img_tensor, (orig_w, orig_h) = self.preprocess_frame(
            frame,
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
            max_det=100 # Limit for real-time
        )
        
        inference_time = time.time() - start_time
        
        pred = predictions[0]
        
        # Properly denormalize boxes
        if len(pred['boxes']) > 0:
            max_val = pred['boxes'].max().item()
            
            # If max value is <= 1.5, boxes are normalized [0,1]
            if max_val <= 1.5:
                # Denormalize from [0,1] to model input size (640x640)
                img_size = self.config['data']['img_size']
                pred['boxes'][:, [0, 2]] *= img_size # x coords
                pred['boxes'][:, [1, 3]] *= img_size # y coords
            
            # Scale from model input size to original frame size
            img_size = self.config['data']['img_size']
            scale_x = orig_w / img_size
            scale_y = orig_h / img_size
            
            pred['boxes'][:, [0, 2]] *= scale_x
            pred['boxes'][:, [1, 3]] *= scale_y
            
            # Clamp to frame boundaries
            pred['boxes'][:, [0, 2]] = pred['boxes'][:, [0, 2]].clamp(0, orig_w)
            pred['boxes'][:, [1, 3]] = pred['boxes'][:, [1, 3]].clamp(0, orig_h)
        
        return pred, inference_time
    
    def draw_detections(self, frame: np.ndarray, predictions: dict, category_names: List[str],
                        show_conf: bool = True):
        """Draw detections on frame with labels ON TOP"""
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        if len(boxes) == 0:
            return frame
        
        # Generate consistent colors
        np.random.seed(42)
        colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
                  for _ in range(len(category_names))]
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            
            # Validate box
            if x2 <= x1 or y2 <= y1:
                continue
            
            color = colors[label]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            label_text = category_names[label]
            if show_conf:
                label_text += f" {score:.2f}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Position label ABOVE box
            label_y1 = max(text_h + 10, y1 - 5) # At least text_h+10 from top
            label_y2 = label_y1 - text_h - 6
            
            # Draw label background
            cv2.rectangle(frame, (x1, label_y2), (x1 + text_w + 8, label_y1), color, -1)
            
            # Draw white border around label
            cv2.rectangle(frame, (x1, label_y2), (x1 + text_w + 8, label_y1), (255, 255, 255), 1)
            
            # Draw text
            cv2.putText(frame, label_text, (x1 + 4, label_y1 - 4), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
        
        return frame
    
    def process_video(self, video_path: str, category_names: List[str], output_path: str = None,
                      conf_threshold: float = 0.25, iou_threshold: float = 0.7, show_fps: bool = True,
                      display: bool = True):
        """Process video file or webcam"""
        # Open video
        if video_path == '0' or video_path == 0:
            cap = cv2.VideoCapture(0)
            self.logger.info("📹 Using webcam")
        else:
            cap = cv2.VideoCapture(video_path)
            self.logger.info(f"📹 Processing: {video_path}")
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        self.logger.info(f"Detecting: {category_names}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"Saving to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict
                predictions, inference_time = self.predict_frame(
                    frame,
                    category_names,
                    conf_threshold,
                    iou_threshold
                )
                
                # Draw detections
                frame = self.draw_detections(frame, predictions, category_names)
                
                # Calculate FPS
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Draw FPS
                if show_fps:
                    fps_text = f"FPS: {avg_fps:.1f}"
                    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Draw detection count
                det_count = len(predictions['boxes'])
                count_text = f"Detections: {det_count}"
                cv2.putText(frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Save frame
                if writer:
                    writer.write(frame)
                
                # Display
                if display:
                    cv2.imshow('YOLO-World Demo', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Stopped by user")
                        break
                    elif key == ord('p'):
                        self.logger.info("Paused - press any key")
                        cv2.waitKey(0)
                
                frame_count += 1
                
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Summary
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            print(f"\n{'='*60}")
            print(f"  Video Processing Complete")
            print(f"{'='*60}")
            print(f"Frames processed: {frame_count}")
            print(f"Average FPS: {avg_fps:.1f}")
            if output_path:
                print(f"Output: {output_path}")
            print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='YOLO-World Video Demo')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, default='0')
    parser.add_argument('--categories', type=str, nargs='+', required=True)
    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--iou_threshold', type=float, default=0.7)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--no_fps', action='store_true')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    demo = VideoDemo(config, args.checkpoint, args.device)
    
    demo.process_video(
        video_path=args.video,
        category_names=args.categories,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        show_fps=not args.no_fps,
        display=not args.no_display
    )
    
    print("Video processing complete!")


if __name__ == '__main__':
    main()