"""
ROI Tree — DFS-based Spatial Hierarchy Search
================================================
Organizes detections into a spatial tree structure:
    Root (frame) → Vehicle nodes → Plate nodes

Uses DFS to search the tree for plates, collect all plates,
and prune low-confidence branches.

Example tree:
    Root (frame)
    ├── Vehicle 1
    │   ├── Plate 1a
    │   └── Plate 1b
    ├── Vehicle 2
    │   └── Plate 2a
    └── Vehicle 3 (no plate)
"""

from typing import List, Optional, Tuple
import numpy as np


class ROINode:
    """Node in the ROI tree — represents one region of interest.
    
    Attributes:
        bbox: [x1, y1, x2, y2] bounding box.
        roi_type: Type of ROI ('frame', 'vehicle', 'plate').
        confidence: Detection confidence.
        children: Child nodes (N-ary tree).
        metadata: Optional extra data.
    """

    def __init__(self, bbox: np.ndarray, roi_type: str = 'unknown',
                 confidence: float = 1.0, metadata: dict = None):
        self.bbox = bbox
        self.roi_type = roi_type
        self.confidence = confidence
        self.children: List['ROINode'] = []
        self.metadata = metadata or {}

    def add_child(self, child: 'ROINode') -> None:
        """Add a child node."""
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"ROINode(type={self.roi_type}, conf={self.confidence:.2f})"


class ROITree:
    """Spatial hierarchy tree for organizing detection results.
    
    Builds a tree from vehicle and plate detections, where plates
    are assigned as children of vehicles they fall within.
    
    DFS is used for:
        - find_best_plate_dfs(): find highest-confidence plate
        - find_all_plates_dfs(): collect all plates
        - prune_low_confidence(): remove weak branches
    """

    def __init__(self, frame_size: Tuple[int, int] = (1920, 1080)):
        """Initialize tree with root node covering the full frame.
        
        Args:
            frame_size: (width, height) of the frame.
        """
        w, h = frame_size
        root_bbox = np.array([0, 0, w, h], dtype=np.float32)
        self.root = ROINode(root_bbox, roi_type='frame')

    def build_from_detections(self,
                              vehicle_boxes: List[np.ndarray],
                              vehicle_scores: List[float],
                              plate_boxes: List[np.ndarray],
                              plate_scores: List[float]) -> None:
        """Build tree from detection results.
        
        Assigns each plate to the vehicle it best fits inside.
        If a plate doesn't fit in any vehicle (overlap < 50%), 
        it's added directly to root.
        
        Args:
            vehicle_boxes: Vehicle bounding boxes.
            vehicle_scores: Vehicle confidence scores.
            plate_boxes: Plate bounding boxes.
            plate_scores: Plate confidence scores.
        """
        # Level 1: Add vehicle nodes under root
        vehicle_nodes: List[ROINode] = []
        for bbox, score in zip(vehicle_boxes, vehicle_scores):
            v_node = ROINode(bbox, roi_type='vehicle', confidence=score)
            self.root.add_child(v_node)
            vehicle_nodes.append(v_node)

        # Level 2: Assign plates to their best-matching vehicle
        for p_bbox, p_score in zip(plate_boxes, plate_scores):
            best_vehicle = None
            best_overlap = 0.0

            for v_node in vehicle_nodes:
                overlap = self._containment_ratio(p_bbox, v_node.bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vehicle = v_node

            p_node = ROINode(p_bbox, roi_type='plate', confidence=p_score)

            if best_vehicle and best_overlap > 0.5:
                best_vehicle.add_child(p_node)
            else:
                # Plate doesn't belong to any vehicle — add to root
                self.root.add_child(p_node)

    def find_best_plate_dfs(self) -> Optional[ROINode]:
        """DFS to find the plate with highest confidence in the tree.
        
        Returns:
            Best plate ROINode, or None.
        """
        best_plate: Optional[ROINode] = None
        best_conf = -1.0

        def _dfs(node: ROINode):
            nonlocal best_plate, best_conf

            if node.roi_type == 'plate' and node.confidence > best_conf:
                best_plate = node
                best_conf = node.confidence

            for child in node.children:
                _dfs(child)

        _dfs(self.root)
        return best_plate

    def find_all_plates_dfs(self) -> List[ROINode]:
        """DFS to collect all plate nodes in the tree.
        
        Returns:
            List of all plate ROINodes.
        """
        plates: List[ROINode] = []

        def _dfs(node: ROINode):
            if node.roi_type == 'plate':
                plates.append(node)
            for child in node.children:
                _dfs(child)

        _dfs(self.root)
        return plates

    def prune_low_confidence(self, min_confidence: float = 0.3) -> int:
        """DFS pruning — remove branches with confidence below threshold.
        
        Args:
            min_confidence: Minimum confidence to keep a node.
        
        Returns:
            Number of nodes pruned.
        """
        pruned_count = 0

        def _prune(node: ROINode) -> bool:
            nonlocal pruned_count

            surviving_children: List[ROINode] = []
            for child in node.children:
                if child.confidence < min_confidence:
                    pruned_count += 1
                else:
                    _prune(child)
                    surviving_children.append(child)

            node.children = surviving_children

        _prune(self.root)
        return pruned_count

    @staticmethod
    def _containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
        """Compute how much of inner box is contained within outer box.
        
        Returns intersection_area / inner_area (1.0 = fully contained).
        
        Args:
            inner: Inner bbox (e.g., plate).
            outer: Outer bbox (e.g., vehicle).
        
        Returns:
            Containment ratio in [0, 1].
        """
        x1 = max(inner[0], outer[0])
        y1 = max(inner[1], outer[1])
        x2 = min(inner[2], outer[2])
        y2 = min(inner[3], outer[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        if inner_area <= 0:
            return 0.0

        return inter_area / inner_area
