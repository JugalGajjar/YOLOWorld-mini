"""
Training Script for YOLO-World
"""

import os
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.coco_dataset import COCODetectionDataset, collate_fn, create_online_vocabulary
from modules.yolo_world import build_yolo_world
from modules.losses import YOLOWorldLoss
from utils.utils import (
    load_config, save_config, setup_logger, set_seed,
    AverageMeter, save_checkpoint, load_checkpoint,
    count_parameters, get_lr, clip_gradients, format_time
)
from utils.device import setup_device_for_training, print_device_info


class Trainer:
    """Trainer class for YOLO-World"""
    
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        
        # Setup directories first
        self.setup_directories()
        self.logger = setup_logger(config['logging']['log_dir'])
        
        # Setup device with auto-detection
        self.device, device_settings = setup_device_for_training(config)
        
        # Adjust config based on device
        if 'recommended_batch_size' in device_settings:
            original_batch = config['train']['batch_size']
            recommended_batch = device_settings['recommended_batch_size']
            if original_batch > recommended_batch and self.device.type in ['cpu', 'mps']:
                self.logger.warning(
                    f"Recommended batch size for {self.device.type.upper()}: {recommended_batch}, "
                    f"but config has {original_batch}. Consider reducing if OOM errors occur."
                )
        
        # Update num_workers based on device
        if self.device.type == 'mps':
            config['data']['num_workers'] = 2 # MPS works better with fewer workers
        
        # Set seed
        set_seed(42)
        
        # Build datasets
        self.train_dataset = self.build_dataset('train')
        self.val_dataset = self.build_dataset('val')
        
        # Worker init function to fix OpenCV multiprocessing issues
        def worker_init_fn(worker_id):
            """Initialize worker process for OpenCV compatibility"""
            import cv2
            # Disable OpenCV threading to avoid conflicts in multiprocessing
            cv2.setNumThreads(0)
        
        # Build dataloaders
        pin_memory = device_settings.get('pin_memory', False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if config['data']['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['val']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if config['data']['num_workers'] > 0 else False
        )
        
        # Build model
        self.logger.info("Building YOLO-World model...")
        self.model = build_yolo_world(config, self.device)
        
        # Log model info
        total_params, trainable_params = count_parameters(self.model)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Build optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
        
        # Build scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['train']['epochs'],
            eta_min=config['train']['learning_rate'] * 0.01
        )
        
        # Build loss
        self.criterion = YOLOWorldLoss(
            num_classes=config['data']['num_classes'],
            box_loss_weight=config['train']['box_loss_weight'],
            cls_loss_weight=config['train']['cls_loss_weight'],
            obj_loss_weight=1.0, # Objectness loss weight
            contrastive_loss_weight=config['train']['contrastive_loss_weight'],
            img_size=config['data']['img_size'],
            reg_max=16
        )
        
        # Training state
        self.start_epoch = 0
        self.best_metric = float('inf')
        
        # Load checkpoint if provided
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['save_dir'], exist_ok=True)
    
    def build_dataset(self, split: str):
        """Build dataset"""
        if split == 'train':
            img_dir = os.path.join(
                self.config['data']['root'],
                self.config['data']['train_img_dir']
            )
            ann_file = os.path.join(
                self.config['data']['root'],
                self.config['data']['train_ann']
            )
            augment = True
        else:
            img_dir = os.path.join(
                self.config['data']['root'],
                self.config['data']['val_img_dir']
            )
            ann_file = os.path.join(
                self.config['data']['root'],
                self.config['data']['val_ann']
            )
            augment = False
        
        dataset = COCODetectionDataset(
            img_dir=img_dir,
            ann_file=ann_file,
            img_size=self.config['data']['img_size'],
            augment=augment
        )
        
        self.logger.info(f"{split} dataset: {len(dataset)} images")
        return dataset
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        box_loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            num_boxes = batch['num_boxes'].to(self.device)
            category_names = batch['category_names']
            
            # Create online vocabulary
            all_categories = self.train_dataset.get_category_names()
            vocabulary, vocab_mapping = create_online_vocabulary(
                batch_categories=category_names,
                all_categories=all_categories,
                max_vocab_size=self.config['train']['vocab_size_per_batch'],
                negative_samples=self.config['train']['negative_samples_per_batch']
            )
            
            # Create label mapping: label_idx -> vocab_idx
            # all_category_names[label_idx] gives the category name
            label_to_vocab = torch.full((self.config['data']['num_classes'],), -1, dtype=torch.long, device=self.device)
            for label_idx in range(self.config['data']['num_classes']):
                cat_name = self.train_dataset.all_category_names[label_idx]
                if cat_name in vocab_mapping:
                    label_to_vocab[label_idx] = vocab_mapping[cat_name]
            
            # Forward pass
            outputs = self.model(images, category_names=vocabulary)
            
            # Compute loss
            targets = {
                'boxes': boxes,
                'labels': labels,
                'num_boxes': num_boxes,
                'label_to_vocab': label_to_vocab, # Add label mapping
            }
            
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            clip_gradients(self.model, max_norm=10.0)
            self.optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item())
            box_loss_meter.update(loss_dict['box_loss'])
            cls_loss_meter.update(loss_dict['cls_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'box': f'{box_loss_meter.avg:.4f}',
                'cls': f'{cls_loss_meter.avg:.4f}',
                'lr': f'{get_lr(self.optimizer):.6f}'
            })
            
            # Log
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f"Epoch [{epoch}][{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"Box: {box_loss_meter.avg:.4f} "
                    f"Cls: {cls_loss_meter.avg:.4f} "
                    f"LR: {get_lr(self.optimizer):.6f}"
                )
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            num_boxes = batch['num_boxes'].to(self.device)
            category_names = batch['category_names']
            
            # Create vocabulary
            all_categories = self.val_dataset.get_category_names()
            vocabulary, vocab_mapping = create_online_vocabulary(
                batch_categories=category_names,
                all_categories=all_categories,
                max_vocab_size=self.config['train']['vocab_size_per_batch']
            )
            
            # Create label mapping
            label_to_vocab = torch.full((self.config['data']['num_classes'],), -1, dtype=torch.long, device=self.device)
            for label_idx in range(self.config['data']['num_classes']):
                cat_name = self.val_dataset.all_category_names[label_idx]
                if cat_name in vocab_mapping:
                    label_to_vocab[label_idx] = vocab_mapping[cat_name]
            
            # Forward pass
            outputs = self.model(images, category_names=vocabulary)
            
            # Compute loss
            targets = {
                'boxes': boxes,
                'labels': labels,
                'num_boxes': num_boxes,
                'label_to_vocab': label_to_vocab,
            }
            
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Update meter
            loss_meter.update(loss.item())
            
            pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
        
        self.logger.info(f"Validation Loss: {loss_meter.avg:.4f}")
        
        return loss_meter.avg
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config['train']['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['train']['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss
            
            if (epoch + 1) % self.config['logging']['save_period'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in {format_time(epoch_time)} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Best: {self.best_metric:.4f}"
            )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best validation loss: {self.best_metric:.4f}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save training checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        filename = f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(
            state,
            self.config['logging']['save_dir'],
            filename,
            is_best
        )
        
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler
        )
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('val_loss', float('inf'))
        
        self.logger.info(f"Resumed from epoch {self.start_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLO-World')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer and train
    trainer = Trainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()