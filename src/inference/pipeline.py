"""
ALPR Pipeline — End-to-End Processing (Composition Pattern)
==============================================================
Combines all modules into a single pipeline:
    Vehicle Detection → Tracking → Counting → Plate Detection → OCR

Uses Composition (has-a) instead of inheritance (is-a) for flexibility.
Each module can be swapped or tested independently.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from ..models.base_detector import Detection
from ..tracking.sort_tracker import SORTTracker
from ..counting.line_counter import LineCrossingCounter
from ..ocr.plate_ocr import PlateOCR


class ALPRPipeline:
    """End-to-end pipeline for Automatic License Plate Recognition.
    
    Components (Composition pattern):
        - VehicleDetector: detect vehicles in frame
        - SORTTracker: track vehicles across frames
        - LineCrossingCounter: count vehicles crossing a line
        - PlateDetector: detect plates within vehicle crops
        - PlateOCR: read plate text
    
    Per-frame flow: detect → track → count → detect plates → OCR
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline from config dict.
        
        Args:
            config: Configuration dictionary with module parameters.
        """
        device = config.get('device', 'cpu')

        # Vehicle detector (pretrained COCO)
        from ..models.vehicle_detector import VehicleDetector
        self.vehicle_detector = VehicleDetector(
            device=device,
            confidence_threshold=config.get('vehicle_conf', 0.5),
        )

        # Plate detector — select based on config
        plate_cfg = config.get('plate_detector', {})
        plate_type = plate_cfg.get('type', 'fasterrcnn')

        if plate_type == 'yolo_plate' or plate_type == 'yolo':
            from ..models.yolo_detector import YOLOPlateDetector
            self.plate_detector = YOLOPlateDetector(
                device=device,
                confidence_threshold=plate_cfg.get('confidence', 0.5),
                model_path=plate_cfg.get('model_path', None),
            )
            print(f"[Pipeline] Loaded YOLOv8 plate detector: {plate_cfg.get('model_path')}")
        else:
            from ..models.plate_detector import PlateDetector
            self.plate_detector = PlateDetector(
                device=device,
                confidence_threshold=plate_cfg.get('confidence', 0.5),
                num_classes= 2,
            )
            print("[Pipeline] Loaded FasterRCNN plate detector")

        # SORT tracker
        tracker_cfg = config.get('tracker', {})
        self.tracker = SORTTracker(
            max_age=tracker_cfg.get('max_age', 3),
            min_hits=tracker_cfg.get('min_hits', 3),
            iou_threshold=tracker_cfg.get('iou_threshold', 0.3),
        )

        # Line crossing counter
        line_cfg = config.get('counting_line', {})
        self.counter = LineCrossingCounter(
            line_start=tuple(line_cfg.get('start', [0, 500])),
            line_end=tuple(line_cfg.get('end', [1920, 500])),
        )

        # Plate OCR
        self.ocr = PlateOCR(
            languages=config.get('ocr_languages', ['en']),
            gpu=(device == 'cuda'),
        )

        # Cache OCR results by track_id to avoid re-reading
        self._plate_cache: Dict[int, str] = {}

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the full pipeline.
        
        Flow: detect vehicles → track → count crossings → detect plates → OCR
        
        Args:
            frame: BGR frame (H, W, 3).
        
        Returns:
            Dict with tracked_vehicles, plate_texts, and counts.
        """
        result: Dict[str, Any] = {
            'tracked_vehicles': [],
            'plate_texts': {},
            'counts': self.counter.get_counts(),
        }

        # Step 1: Detect vehicles
        vehicle_dets: List[Detection] = self.vehicle_detector.detect(frame)

        if not vehicle_dets:
            return result

        # Convert to SORT format: [x1, y1, x2, y2, score]
        det_array = np.array([
            d.bbox + [d.confidence] for d in vehicle_dets
        ], dtype=np.float32)

        # Step 2: Update tracker
        tracks = self.tracker.update(det_array)

        # Step 3-5: Process each track
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            bbox = [x1, y1, x2, y2]

            # Compute centroid
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Step 3: Check line crossing
            crossed = self.counter.update(track_id, (cx, cy))

            # Step 4-5: Plate detection + OCR (cached per track)
            plate_text = self._plate_cache.get(track_id, "")

            if not plate_text:
                # Crop vehicle region from frame
                ix1, iy1, ix2, iy2 = map(int, bbox)
                ix1 = max(0, ix1)
                iy1 = max(0, iy1)
                ix2 = min(frame.shape[1], ix2)
                iy2 = min(frame.shape[0], iy2)
                vehicle_crop = frame[iy1:iy2, ix1:ix2]

                if vehicle_crop.size > 0:
                    plate_dets = self.plate_detector.detect(vehicle_crop)

                    if plate_dets:
                        best_plate = max(plate_dets, key=lambda d: d.confidence)
                        px1, py1, px2, py2 = map(int, best_plate.bbox)
                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        if plate_crop.size > 0:
                            text, conf = self.ocr.read_plate(plate_crop)
                            if text and conf > 0.3:
                                plate_text = text
                                self._plate_cache[track_id] = text

            result['tracked_vehicles'].append({
                'track_id': track_id,
                'bbox': bbox,
                'centroid': (cx, cy),
                'crossed': crossed,
            })

            if plate_text:
                result['plate_texts'][track_id] = plate_text

        result['counts'] = self.counter.get_counts()

        return result

    def reset(self) -> None:
        """Reset all pipeline state."""
        self.tracker.reset()
        self.counter.reset()
        self._plate_cache.clear()
