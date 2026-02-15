"""
YOLOv8 Training Pipeline
============================
Custom training wrapper around Ultralytics YOLOv8 that integrates
with our project's logging, checkpointing, and evaluation infrastructure.

Features:
    - Automatic COCO → YOLO format conversion
    - Custom callback integration for ExperimentLogger
    - Model comparison support (FasterRCNN vs YOLO)
    - Training summary and export
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class YOLOTrainer:
    """Training orchestrator for YOLOv8 models.

    Handles the full training lifecycle:
        1. Dataset preparation (COCO → YOLO conversion)
        2. Model initialization
        3. Training with advanced augmentation
        4. Evaluation on validation/test sets
        5. Results logging and model export

    Args:
        config: Training configuration dictionary.
        device: Computing device ('cpu' or 'cuda').
    """

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.results = None
        self.data_yaml = None

    def prepare_dataset(self, coco_dir: str = 'data/coco',
                         yolo_dir: str = 'data/yolo') -> str:
        """Convert COCO dataset to YOLO format.

        Args:
            coco_dir: Path to COCO-format dataset.
            yolo_dir: Output path for YOLO-format dataset.

        Returns:
            Path to generated data.yaml file.
        """
        from ..data.yolo_dataset import prepare_yolo_dataset

        self.data_yaml = prepare_yolo_dataset(coco_dir, yolo_dir)
        return self.data_yaml

    def train(self, data_yaml: Optional[str] = None) -> Dict[str, Any]:
        """Run YOLOv8 training.

        Args:
            data_yaml: Path to data.yaml. If None, uses prepared dataset.

        Returns:
            Dict with training results, metrics, and paths.
        """
        from ..models.yolo_detector import YOLOPlateDetector

        data_yaml = data_yaml or self.data_yaml
        if not data_yaml:
            raise ValueError("No data.yaml provided. Run prepare_dataset() first.")

        # Extract config values
        model_cfg = self.config.get('model', {})
        train_cfg = self.config.get('training', {})
        aug_cfg = train_cfg.get('augmentation', {})

        variant = model_cfg.get('variant', 'yolov8n')

        print("\n" + "🚀" * 20)
        print("  STARTING YOLOv8 TRAINING")
        print("🚀" * 20)

        # Initialize detector
        detector = YOLOPlateDetector(
            device=self.device,
            model_variant=variant,
            confidence_threshold=0.5,
        )

        start_time = time.time()

        # Train with config parameters
        train_result = detector.train_model(
            data_yaml=data_yaml,
            epochs=train_cfg.get('epochs', 50),
            batch=train_cfg.get('batch_size', 16),
            imgsz=train_cfg.get('imgsz', 640),
            optimizer=train_cfg.get('optimizer', 'AdamW'),
            lr=train_cfg.get('lr', 0.01),
            lrf=train_cfg.get('lrf', 0.01),
            warmup_epochs=train_cfg.get('warmup_epochs', 3),
            patience=train_cfg.get('patience', 20),
            save_period=train_cfg.get('save_period', 10),
            project='runs/yolo',
            name='train',
            # Augmentation
            mosaic=aug_cfg.get('mosaic', 1.0),
            mixup=aug_cfg.get('mixup', 0.15),
            degrees=aug_cfg.get('degrees', 10.0),
            fliplr=aug_cfg.get('fliplr', 0.5),
            flipud=aug_cfg.get('flipud', 0.0),
            erasing=aug_cfg.get('erasing', 0.3),
            copy_paste=aug_cfg.get('copy_paste', 0.1),
        )

        elapsed = time.time() - start_time

        # Evaluate on validation set
        print("\n📊 Evaluating on validation set...")
        eval_results = detector.evaluate_model(data_yaml)

        # Compile results
        self.results = {
            'training_time': elapsed,
            'training_time_str': f"{elapsed / 60:.1f} minutes",
            'best_model': train_result.get('best_model', ''),
            'last_model': train_result.get('last_model', ''),
            'metrics': {
                'mAP50': eval_results.get('mAP50', 0),
                'mAP50_95': eval_results.get('mAP50_95', 0),
                'precision': eval_results.get('precision', 0),
                'recall': eval_results.get('recall', 0),
            },
            'config': {
                'variant': variant,
                'epochs': train_cfg.get('epochs', 50),
                'batch_size': train_cfg.get('batch_size', 16),
                'imgsz': train_cfg.get('imgsz', 640),
                'optimizer': train_cfg.get('optimizer', 'AdamW'),
            }
        }

        self._print_results()

        return self.results

    def _print_results(self) -> None:
        """Display training results summary."""
        if not self.results:
            return

        metrics = self.results['metrics']

        print("\n" + "=" * 60)
        print("  📊 YOLOv8 TRAINING RESULTS")
        print("=" * 60)
        print(f"  ⏱️  Training Time:  {self.results['training_time_str']}")
        print(f"  📦  Best Model:     {self.results['best_model']}")
        print()
        print(f"  🎯  mAP@0.5:       {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)")
        print(f"  🎯  mAP@0.5:0.95:  {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.1f}%)")
        print(f"  ✅  Precision:      {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"  ✅  Recall:         {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print("=" * 60)

    def compare_with_baseline(self, baseline_results: Dict) -> None:
        """Print comparison between YOLO and baseline (FasterRCNN) results.

        Args:
            baseline_results: Dict with baseline metrics (mAP, precision, recall).
        """
        if not self.results:
            print("⚠ No YOLO results available. Train first.")
            return

        yolo = self.results['metrics']
        base = baseline_results

        print("\n" + "=" * 70)
        print("  📊 MODEL COMPARISON: YOLOv8 vs FasterRCNN-MobileNetV3")
        print("=" * 70)
        print(f"  {'Metric':<20} {'FasterRCNN':>15} {'YOLOv8':>15} {'Improvement':>15}")
        print("-" * 70)

        for metric in ['mAP50', 'precision', 'recall']:
            base_val = base.get(metric, 0)
            yolo_val = yolo.get(metric, 0)
            diff = yolo_val - base_val
            arrow = "↑" if diff > 0 else "↓"

            print(f"  {metric:<20} {base_val:>14.4f} {yolo_val:>14.4f} "
                  f"{arrow} {abs(diff):>12.4f}")

        print("=" * 70)
