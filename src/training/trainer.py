"""
Custom Training Loop for Object Detection
============================================
Implements a full training pipeline without PyTorch Lightning:
- SGD with Momentum optimizer
- Warmup + Cosine Annealing learning rate schedule
- Mixed Precision Training (AMP) for GPU efficiency
- Gradient Clipping to prevent explosion
- Checkpointing with resume support
- Early Stopping based on validation mAP
"""

import os
import time
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from .evaluator import Evaluator
from ..utils.logger import ExperimentLogger


class Trainer:
    """Training loop for object detection models.
    
    Each epoch: train -> evaluate -> checkpoint -> early_stop check
    
    Args:
        model: PyTorch model (e.g., FasterRCNN).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        config: Config dictionary from YAML.
        device: 'cpu' or 'cuda'.
    """

    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Hyperparameters from config
        train_cfg = config.get('training', {})
        
        self.epochs = train_cfg.get('epochs', 50)
        self.lr = train_cfg.get('lr', 0.005)
        self.momentum = train_cfg.get('momentum', 0.9)
        self.weight_decay = train_cfg.get('weight_decay', 5e-4)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 3)
        self.clip_grad_norm = train_cfg.get('clip_grad_norm', 10.0)
        self.patience = train_cfg.get('patience', 10)

        # SGD optimizer with momentum and L2 regularization
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # GradScaler for mixed precision (auto loss scaling)
        self.scaler = GradScaler()

        # Training state
        self.best_map = 0.0
        self.epochs_no_improve = 0
        self.start_epoch = 0

        # Evaluator and logger
        model_cfg = config.get('model', {})
        self.evaluator = Evaluator(device=device)
        self.num_classes = model_cfg.get('num_classes', 2)
        
        log_cfg = config.get('logging', {})
        log_dir = log_cfg.get('log_dir', 'runs/train')
        self.logger = ExperimentLogger(log_dir)

        # Checkpoint directory
        self.checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Resume from checkpoint if specified
        resume_path = train_cfg.get('resume_from', None)
        if resume_path and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    def _compute_lr(self, epoch: int) -> float:
        """Compute learning rate with warmup + cosine annealing.
        
        Warmup: linearly ramp up from ~0 to lr_base over warmup_epochs.
        Cosine: decay from lr_base to lr_min following a cosine curve.
        
        Args:
            epoch: Current epoch (0-based).
        
        Returns:
            Learning rate for this epoch.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup: avoids gradient explosion in early epochs
            return self.lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: smooth decay from lr to lr_min
            t = epoch - self.warmup_epochs
            T = self.epochs - self.warmup_epochs
            min_lr = self.lr * 0.01  # Minimum lr = 1% of initial
            return min_lr + 0.5 * (self.lr - min_lr) * (1 + math.cos(math.pi * t / T))

    def _update_lr(self, epoch: int) -> None:
        """Update optimizer learning rate for given epoch."""
        new_lr = self._compute_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train_one_epoch(self, epoch: int) -> float:
        """Train model for one epoch over the full training set.
        
        Uses mixed precision (autocast + GradScaler) on GPU and
        gradient clipping to stabilize training.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            Average loss over the epoch.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass with mixed precision
            with autocast(enabled=(self.device == 'cuda')):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(losses).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.clip_grad_norm
            )

            # Optimizer step (skipped if gradients contain inf/nan)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses.item()
            num_batches += 1

            # Print progress every 20 batches
            if batch_idx % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch [{epoch}] Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {losses.item():.4f} LR: {current_lr:.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self) -> Dict[str, Any]:
        """Main training loop — orchestrates the full training process.
        
        For each epoch: update LR -> train -> evaluate -> checkpoint -> early stop check
        
        Returns:
            Dict with best_mAP and total training time.
        """
        print(f"\n{'='*60}")
        print(f"Starting Training — {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"LR: {self.lr}, Momentum: {self.momentum}")
        print(f"Warmup: {self.warmup_epochs} epochs")
        print(f"Early Stopping Patience: {self.patience}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # Step 1: Update learning rate
            self._update_lr(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Step 2: Train one epoch
            avg_loss = self.train_one_epoch(epoch)

            # Step 3: Evaluate on validation set
            eval_results = self.evaluator.evaluate(
                self.model, self.val_loader, self.num_classes
            )
            current_map = eval_results.get('mAP@0.5', 0.0)

            # Step 4: Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.log_scalar('train/loss', avg_loss, epoch)
            self.logger.log_scalar('train/lr', current_lr, epoch)
            self.logger.log_scalar('val/mAP@0.5', current_map, epoch)

            print(f"\nEpoch [{epoch}/{self.epochs}] — "
                  f"Loss: {avg_loss:.4f} | mAP@0.5: {current_map:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            # Step 5: Save checkpoint if improved
            if current_map > self.best_map:
                self.best_map = current_map
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ★ New best mAP@0.5: {self.best_map:.4f} — Saved!")
            else:
                self.epochs_no_improve += 1
                print(f"  ✗ No improvement ({self.epochs_no_improve}/{self.patience})")

            # Step 6: Early stopping check
            if self.epochs_no_improve >= self.patience:
                print(f"\n⚠ Early Stopping! No improvement for {self.patience} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete! Total time: {total_time:.1f}s")
        print(f"Best mAP@0.5: {self.best_map:.4f}")
        print(f"{'='*60}\n")

        self.logger.close()

        return {
            'best_mAP@0.5': self.best_map,
            'total_time': total_time,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint (model + optimizer + scaler state).
        
        Args:
            epoch: Current epoch.
            is_best: If True, also save as 'best.pth'.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_map': self.best_map,
        }

        path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)

    def _load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint file.
        
        Args:
            path: Path to checkpoint .pth file.
        """
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)

        print(f"Resumed from epoch {checkpoint['epoch']}, mAP: {self.best_map:.4f}")
