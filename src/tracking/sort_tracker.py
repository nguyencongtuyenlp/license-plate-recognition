"""
SORT Tracker — Simple Online and Realtime Tracking
=====================================================
Combines Kalman Filter for state prediction and the
Hungarian Algorithm for detection-to-track association.

State vector: [cx, cy, s, r, dx, dy, ds]
    cx, cy = bounding box center
    s = scale (area = w * h)
    r = aspect ratio (w / h)
    dx, dy, ds = velocities (constant velocity model)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..training.metrics import compute_iou


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert bbox [x1, y1, x2, y2] to Kalman state [cx, cy, s, r].
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2].
    
    Returns:
        State vector [cx, cy, s, r] as column vector (4, 1).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h                    # area
    r = w / (h + 1e-6)           # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, s, r] back to bbox [x1, y1, x2, y2].
    
    Args:
        z: Kalman state [cx, cy, s, r].
    
    Returns:
        Bounding box [x1, y1, x2, y2].
    """
    cx, cy, s, r = z.flatten()
    w = np.sqrt(max(s * r, 1.0))
    h = s / (w + 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ])


class KalmanBoxTracker:
    """Kalman Filter tracker for a single bounding box.
    
    Uses a constant velocity model with 7D state:
    [cx, cy, s, r, dx, dy, ds] where velocities are estimated
    from consecutive detections.
    
    The transition matrix F encodes: new_position = old_position + velocity.
    Observation matrix H extracts [cx, cy, s, r] from the full state.
    """
    _id_counter = 0  # Global ID counter shared across all trackers

    def __init__(self, bbox: np.ndarray):
        """Initialize tracker with first detection.
        
        Args:
            bbox: Initial detection [x1, y1, x2, y2].
        """
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter

        # Initialize 7D state, 4D measurement Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix (only observe position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Measurement noise — scale/ratio measurements are noisier
        self.kf.R[2:, 2:] *= 10.0

        # Initial state covariance — high uncertainty for velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise — velocities change slowly
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Set initial state from first detection (velocities start at 0)
        self.kf.x[:4] = _bbox_to_z(bbox)

        # Track lifecycle management
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history: List[np.ndarray] = []

    def predict(self) -> np.ndarray:
        """Predict next position using Kalman prediction step.
        
        Returns:
            Predicted bbox [x1, y1, x2, y2].
        """
        # Prevent negative area
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        predicted_bbox = _z_to_bbox(self.kf.x[:4])
        self.history.append(predicted_bbox)
        return predicted_bbox

    def update(self, bbox: np.ndarray) -> None:
        """Update state with a matched detection (Kalman update step).
        
        Args:
            bbox: Matched detection [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []
        self.kf.update(_bbox_to_z(bbox))

    def get_state(self) -> np.ndarray:
        """Return current estimated bbox [x1, y1, x2, y2]."""
        return _z_to_bbox(self.kf.x[:4])


class SORTTracker:
    """SORT: Simple Online and Realtime Tracking.
    
    Per-frame algorithm:
        1. PREDICT: all trackers predict next position
        2. ASSOCIATE: match predictions to detections (Hungarian Algorithm)
        3. UPDATE: matched trackers get measurement update
        4. CREATE: unmatched detections become new trackers
        5. DELETE: trackers not matched for too long are removed
    
    Args:
        max_age: Max frames a tracker can survive without matching.
        min_hits: Min consecutive hits before a track is confirmed.
        iou_threshold: Min IoU for a valid match.
    """

    def __init__(self, max_age: int = 3, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame of detections.
        
        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, score].
        
        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id] for active tracks.
        """
        self.frame_count += 1

        # Step 1: Predict — get predicted positions for all trackers
        predicted_boxes = []
        trackers_to_delete = []

        for i, tracker in enumerate(self.trackers):
            pred = tracker.predict()
            if np.any(np.isnan(pred)):
                trackers_to_delete.append(i)
            else:
                predicted_boxes.append(pred)

        # Remove diverged trackers (iterate in reverse to preserve indices)
        for i in reversed(trackers_to_delete):
            self.trackers.pop(i)

        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # Step 2: Associate — match detections to predictions
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections[:, :4] if len(detections) > 0 else np.empty((0, 4)),
            predicted_boxes,
        )

        # Step 3: Update — matched trackers receive new measurements
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx, :4])

        # Step 4: Create — new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_tracker = KalmanBoxTracker(detections[det_idx, :4])
            self.trackers.append(new_tracker)

        # Step 5: Delete old trackers and collect results
        active_trackers = []
        results = []

        for tracker in self.trackers:
            if tracker.time_since_update <= self.max_age:
                active_trackers.append(tracker)
                
                # Only return confirmed tracks (enough consecutive hits)
                if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = tracker.get_state()
                    results.append(np.concatenate([bbox, [tracker.id]]))

        self.trackers = active_trackers

        return np.array(results) if results else np.empty((0, 5))

    def _associate(self, detections: np.ndarray,
                   predictions: np.ndarray
                   ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to predictions using the Hungarian Algorithm.
        
        Cost matrix = 1 - IoU (minimizing cost = maximizing IoU).
        
        Returns:
            Tuple of (matched_pairs, unmatched_det_indices, unmatched_trk_indices).
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))

        # Build IoU matrix
        num_dets = len(detections)
        num_preds = len(predictions)
        iou_matrix = np.zeros((num_dets, num_preds))

        for d in range(num_dets):
            for p in range(num_preds):
                iou_matrix[d, p] = compute_iou(detections[d], predictions[p])

        # Solve assignment (minimize cost = 1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by IoU threshold
        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_trks = list(range(num_preds))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.remove(r)
                unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    def reset(self) -> None:
        """Reset all trackers and counters."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0
