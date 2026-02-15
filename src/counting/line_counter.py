"""
Line-Crossing Counter — Vehicle Counting via Virtual Line
============================================================
Counts vehicles crossing a virtual line using centroid tracking.
Uses cross product to determine which side of the line a point
is on, and detects crossings when the sign changes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class LineCrossingCounter:
    """Counts vehicles crossing a virtual line, with direction tracking.
    
    Algorithm:
        1. For each tracked object, store previous centroid position
        2. Compute cross product with the counting line for both positions
        3. If sign changes → object has crossed the line
        4. Direction determined by sign transition:
           positive → negative = "in", negative → positive = "out"
    
    Args:
        line_start: (x, y) start point of the counting line.
        line_end: (x, y) end point of the counting line.
    """

    def __init__(self, line_start: Tuple[int, int],
                 line_end: Tuple[int, int]):
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)

        # Previous centroid positions per track ID
        self._prev_positions: Dict[int, np.ndarray] = {}

        # Counting state
        self.count_in = 0
        self.count_out = 0
        self.total_count = 0
        self.counted_ids: set = set()  # Already-counted track IDs

        # Per-class vehicle counts
        self.class_counts: Dict[str, int] = {}

    def _cross_product_sign(self, point: np.ndarray) -> float:
        """Compute 2D cross product to determine which side of the line a point is on.
        
        cross = (B - A) x (P - A)
        Positive = left of line, Negative = right of line.
        
        Args:
            point: Position (x, y).
        
        Returns:
            Cross product value (positive = left, negative = right).
        """
        line_vec = self.line_end - self.line_start
        point_vec = point - self.line_start
        cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
        return cross

    def update(self, track_id: int, centroid: Tuple[float, float],
               class_name: str = "vehicle") -> bool:
        """Check if a tracked object has crossed the counting line.
        
        Args:
            track_id: Unique track ID.
            centroid: Current center position (cx, cy).
            class_name: Vehicle class name.
        
        Returns:
            True if the object crossed the line on this update.
        """
        if track_id in self.counted_ids:
            return False

        current_pos = np.array(centroid, dtype=np.float32)
        current_sign = self._cross_product_sign(current_pos)

        if track_id in self._prev_positions:
            prev_pos = self._prev_positions[track_id]
            prev_sign = self._cross_product_sign(prev_pos)

            # Sign change = line crossing detected
            if prev_sign * current_sign < 0:
                self.total_count += 1
                self.counted_ids.add(track_id)

                # Determine crossing direction
                if prev_sign > 0 and current_sign < 0:
                    self.count_in += 1
                else:
                    self.count_out += 1

                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

                self._prev_positions[track_id] = current_pos
                return True

        self._prev_positions[track_id] = current_pos
        return False

    def get_counts(self) -> Dict[str, any]:
        """Get current counting results.
        
        Returns:
            Dict with total_count, count_in, count_out, counts_by_class.
        """
        return {
            "total_count": self.total_count,
            "count_in": self.count_in,
            "count_out": self.count_out,
            "counts_by_class": dict(self.class_counts),
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.count_in = 0
        self.count_out = 0
        self.total_count = 0
        self.counted_ids.clear()
        self.class_counts.clear()
        self._prev_positions.clear()

    @property
    def line_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return line endpoints as integer tuples (for drawing)."""
        return (
            tuple(self.line_start.astype(int)),
            tuple(self.line_end.astype(int)),
        )
