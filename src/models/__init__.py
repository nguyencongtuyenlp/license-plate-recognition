"""
models — Detection models.

Contains:
    - BaseDetector (ABC): Abstract interface for all detectors
    - DetectorFactory: Factory pattern for creating detectors by name
    - PlateDetector: FasterRCNN-MobileNetV3 for license plate detection
    - VehicleDetector: Pretrained COCO model for vehicle detection
"""
