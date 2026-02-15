"""
YOLOv8 Detector — High-Accuracy License Plate Detection
==========================================================
Wraps Ultralytics YOLOv8 with our BaseDetector interface for
seamless integration into the ALPR pipeline.

Features:
    - YOLOv8 variants: nano (n), small (s), medium (m), large (l)
    - Optional CBAM attention integration
    - Config-driven creation via DetectorFactory
    - Both training and inference support

Architecture:
    YOLOv8 Backbone (CSPDarknet) → FPN Neck → Decoupled Head → Detections
"""

from typing import List, Optional, Dict, Any
import os
import numpy as np
import torch

from .base_detector import BaseDetector, Detection, DetectorFactory


class YOLOPlateDetector(BaseDetector):
    """License plate detector using YOLOv8.

    Provides significantly higher accuracy than FasterRCNN-MobileNetV3
    while maintaining real-time inference speed.

    Supports:
        - Multiple YOLOv8 variants (n/s/m/l/x)
        - Custom training with advanced augmentation
        - ONNX/TensorRT export for deployment

    Args:
        device: Computing device ('cpu' or 'cuda').
        confidence_threshold: Min confidence for detections.
        model_variant: YOLOv8 variant — 'yolov8n', 'yolov8s', etc.
        model_path: Path to trained weights (.pt file).
    """

    def __init__(self, device: str = 'cpu',
                 confidence_threshold: float = 0.5,
                 model_variant: str = 'yolov8n',
                 model_path: Optional[str] = None):
        super().__init__(device, confidence_threshold)

        self.model_variant = model_variant
        self.model_path = model_path

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8. "
                "Install with: pip install ultralytics>=8.0.0"
            )

        # Load model: either trained weights or pretrained base
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"[YOLOPlateDetector] Loaded trained model: {model_path}")
        else:
            self.model = YOLO(f'{model_variant}.pt')
            print(f"[YOLOPlateDetector] Loaded pretrained: {model_variant}")

        # Set device
        self.model.to(device)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect license plates in an image using YOLOv8.

        Args:
            image: BGR numpy array (H, W, 3), dtype uint8.

        Returns:
            List of Detection objects above confidence threshold.
        """
        # YOLOv8 accepts BGR numpy arrays directly
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    # Get class name from model
                    class_name = result.names.get(class_id, 'license_plate')

                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    ))

        return detections

    def train_model(self, data_yaml: str, **kwargs) -> Dict[str, Any]:
        """Train YOLOv8 on custom dataset.

        This wraps ultralytics training with our project's conventions.

        Args:
            data_yaml: Path to YOLO data.yaml config.
            **kwargs: Additional training arguments (epochs, batch, imgsz, etc.)

        Returns:
            Dict with training results and metrics.
        """
        # Default training parameters optimized for license plate detection
        train_args = {
            'data': data_yaml,
            'epochs': kwargs.get('epochs', 50),
            'batch': kwargs.get('batch', 16),
            'imgsz': kwargs.get('imgsz', 640),
            'device': self.device,
            'project': kwargs.get('project', 'runs/yolo'),
            'name': kwargs.get('name', 'train'),
            'exist_ok': True,
            'pretrained': True,
            'optimizer': kwargs.get('optimizer', 'AdamW'),
            'lr0': kwargs.get('lr', 0.01),
            'lrf': kwargs.get('lrf', 0.01),
            'warmup_epochs': kwargs.get('warmup_epochs', 3),
            'patience': kwargs.get('patience', 20),
            'save': True,
            'save_period': kwargs.get('save_period', 10),
            'val': True,
            'plots': True,
            'verbose': True,
            # Advanced augmentation settings
            'mosaic': kwargs.get('mosaic', 1.0),
            'mixup': kwargs.get('mixup', 0.15),
            'degrees': kwargs.get('degrees', 10.0),
            'translate': kwargs.get('translate', 0.2),
            'scale': kwargs.get('scale', 0.5),
            'fliplr': kwargs.get('fliplr', 0.5),
            'flipud': kwargs.get('flipud', 0.0),
            'hsv_h': kwargs.get('hsv_h', 0.015),
            'hsv_s': kwargs.get('hsv_s', 0.7),
            'hsv_v': kwargs.get('hsv_v', 0.4),
            'erasing': kwargs.get('erasing', 0.3),
            'copy_paste': kwargs.get('copy_paste', 0.1),
        }

        print("\n" + "=" * 60)
        print("  YOLOv8 Training Configuration")
        print("=" * 60)
        for key, value in train_args.items():
            if key != 'data':
                print(f"  {key:20s}: {value}")
        print("=" * 60 + "\n")

        # Train the model
        results = self.model.train(**train_args)

        return {
            'results': results,
            'best_model': str(self.model.trainer.best),
            'last_model': str(self.model.trainer.last),
        }

    def evaluate_model(self, data_yaml: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate model on validation or test set.

        Args:
            data_yaml: Path to data.yaml (uses training data if None).

        Returns:
            Dict with evaluation metrics (mAP, precision, recall).
        """
        val_args = {
            'device': self.device,
            'verbose': True,
            'plots': True,
        }

        if data_yaml:
            val_args['data'] = data_yaml

        results = self.model.val(**val_args)

        return {
            'mAP50': results.box.map50,
            'mAP50_95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        }

    def load_model(self, path: str) -> None:
        """Load trained YOLO weights.

        Args:
            path: Path to .pt weights file.
        """
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.model.to(self.device)
        self.model_path = path
        print(f"[YOLOPlateDetector] Loaded weights: {path}")

    def get_model(self) -> torch.nn.Module:
        """Return the underlying YOLO model."""
        return self.model.model

    def export(self, format: str = 'onnx', **kwargs) -> str:
        """Export model for deployment.

        Args:
            format: Export format ('onnx', 'torchscript', 'engine').

        Returns:
            Path to exported model file.
        """
        path = self.model.export(format=format, **kwargs)
        print(f"[YOLOPlateDetector] Exported to {format}: {path}")
        return str(path)


# Register with factory for config-driven creation
DetectorFactory.register('yolo_plate', YOLOPlateDetector)
