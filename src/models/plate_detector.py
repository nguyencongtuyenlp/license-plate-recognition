"""
Plate Detector — FasterRCNN-MobileNetV3 for License Plate Detection
=====================================================================
Fine-tuned FasterRCNN with MobileNetV3-Large backbone + FPN
for detecting license plates.

Uses transfer learning: pretrained COCO backbone with a custom
classification head for 2 classes (background + license_plate).
"""

from typing import List, Optional
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base_detector import BaseDetector, Detection, DetectorFactory


class PlateDetector(BaseDetector):
    """License plate detector using FasterRCNN-MobileNetV3-FPN.
    
    Implements BaseDetector interface with transfer learning:
    loads COCO-pretrained model and replaces the classification head
    for license plate detection (2 classes: background + plate).
    
    Args:
        device: 'cpu' or 'cuda'.
        confidence_threshold: Minimum confidence score (default 0.5).
        num_classes: Number of classes (2 = background + license_plate).
    """

    def __init__(self, device: str = 'cpu',
                 confidence_threshold: float = 0.5,
                 num_classes: int = 2):
        super().__init__(device, confidence_threshold)

        self.num_classes = num_classes

        # Load pretrained FasterRCNN-MobileNetV3 (COCO weights)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

        # Replace classification head for our number of classes
        # Original head predicts 91 COCO classes, we need num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        self.model.to(self.device)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect license plates in an image.
        
        Args:
            image: BGR numpy array (H, W, 3), dtype uint8.
        
        Returns:
            List of Detection objects above confidence threshold.
        """
        self.model.eval()

        # Preprocess: BGR -> RGB -> float tensor [0, 1]
        img_rgb = image[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])

        # Filter detections by confidence threshold
        detections: List[Detection] = []
        output = outputs[0]

        for i in range(len(output['boxes'])):
            score = output['scores'][i].item()

            if score >= self.confidence_threshold:
                box = output['boxes'][i].cpu().numpy().tolist()
                label = output['labels'][i].item()

                detections.append(Detection(
                    bbox=box,
                    confidence=score,
                    class_id=label,
                    class_name='license_plate',
                ))

        return detections

    def load_model(self, path: str) -> None:
        """Load model weights from checkpoint file.
        
        Args:
            path: Path to .pth checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print(f"[PlateDetector] Loaded weights from: {path}")

    def get_model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model (for training/evaluation)."""
        return self.model


# Register with factory for config-driven creation
DetectorFactory.register('plate', PlateDetector)
