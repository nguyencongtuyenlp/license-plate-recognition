"""
Base Detector Classes and Factory Pattern
==========================================
Defines the common interface for all detectors and a factory
for creating detector instances by name.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np
import torch


@dataclass
class Detection:
    """Data structure for a single detection result.
    
    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        confidence: Confidence score [0, 1].
        class_id: Class ID (e.g., 1 = license_plate).
        class_name: Class name string.
    """
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str = ""


class BaseDetector(ABC):
    """Abstract base class for all detectors.
    
    Defines the common interface that all detector subclasses must implement.
    Cannot be instantiated directly - subclasses must implement all abstract methods.
    
    Args:
        device: Computing device ('cpu' or 'cuda').
        confidence_threshold: Minimum confidence to keep detections.
    """

    def __init__(self, device: str = 'cpu',
                 confidence_threshold: float = 0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in an image.
        
        Args:
            image: Input image in BGR format (H, W, 3) from OpenCV.
        
        Returns:
            List of Detection objects.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model weights from checkpoint.
        
        Args:
            path: Path to checkpoint file (.pth).
        """
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return the underlying PyTorch model.
        
        Returns:
            torch.nn.Module instance.
        """
        pass


class DetectorFactory:
    """Factory for creating detector instances by name.
    
    Usage:
        # Register a detector class
        DetectorFactory.register('plate', PlateDetector)
        
        # Create instance by name
        detector = DetectorFactory.create('plate', device='cuda')
    
    Benefits:
        - Client code doesn't need to know specific detector classes
        - Easy to add new detectors (just register them)
        - Config-driven: select detector from YAML config
    """

    _registry: Dict[str, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, name: str, detector_class: Type[BaseDetector]) -> None:
        """Register a detector class.
        
        Args:
            name: Name for lookup (e.g., 'plate', 'vehicle').
            detector_class: Detector class (not instance).
        """
        cls._registry[name] = detector_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDetector:
        """Create detector instance by name.
        
        Args:
            name: Registered detector name.
            **kwargs: Arguments passed to constructor.
        
        Returns:
            Detector instance.
        
        Raises:
            ValueError: If name not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Detector '{name}' not registered. "
                f"Available: {available}"
            )
        
        detector_class = cls._registry[name]
        return detector_class(**kwargs)

    @classmethod
    def list_detectors(cls) -> List[str]:
        """List all registered detector names."""
        return list(cls._registry.keys())
