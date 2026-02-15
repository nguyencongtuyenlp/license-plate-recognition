"""
Vehicle Detector — Pretrained COCO FasterRCNN
================================================
Detects vehicles using a COCO-pretrained FasterRCNN model.
No additional training needed — filters COCO predictions
for vehicle classes only (car, motorcycle, bus, truck).
"""

from typing import Dict, List, Set
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from .base_detector import BaseDetector, Detection, DetectorFactory


class VehicleDetector(BaseDetector):
    """Vehicle detector using pretrained FasterRCNN on COCO.
    
    Unlike PlateDetector, this doesn't modify the model head.
    It uses the full 91-class COCO model and simply filters
    results for vehicle classes only.
    
    Args:
        device: 'cpu' or 'cuda'.
        confidence_threshold: Minimum confidence score (default 0.5).
    """

    # COCO class IDs for vehicles (O(1) lookup with set)
    VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}

    COCO_NAMES: Dict[int, str] = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }

    def __init__(self, device: str = 'cpu',
                 confidence_threshold: float = 0.5):
        super().__init__(device, confidence_threshold)

        # Use pretrained COCO model as-is (no head replacement)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect vehicles in an image.
        
        Args:
            image: BGR numpy array (H, W, 3).
        
        Returns:
            List of Detection objects for vehicles only.
        """
        self.model.eval()

        # Preprocess: BGR -> RGB -> float tensor
        img_rgb = image[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])

        # Filter for vehicle classes only
        detections: List[Detection] = []
        output = outputs[0]

        for i in range(len(output['boxes'])):
            label = output['labels'][i].item()
            score = output['scores'][i].item()

            if label in self.VEHICLE_CLASS_IDS and score >= self.confidence_threshold:
                box = output['boxes'][i].cpu().numpy().tolist()
                class_name = self.COCO_NAMES.get(label, 'vehicle')

                detections.append(Detection(
                    bbox=box,
                    confidence=score,
                    class_id=label,
                    class_name=class_name,
                ))

        return detections

    def load_model(self, path: str) -> None:
        """Load model weights (usually not needed for pretrained)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def get_model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model."""
        return self.model


# Register with factory
DetectorFactory.register('vehicle', VehicleDetector)
