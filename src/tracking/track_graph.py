"""
Track Graph — DFS-based Trajectory Reconstruction
=====================================================
Builds a graph from detections across frames and uses
DFS to find complete trajectories (paths through the graph).

Useful for reconnecting broken tracks when SORT loses
a target due to occlusion.

Nodes: individual detections (frame_id, detection_id)
Edges: IoU-based similarity between detections in nearby frames
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..training.metrics import compute_iou


class TrackNode:
    """A node in the tracking graph, representing one detection.
    
    Attributes:
        frame_id: Frame number.
        detection_id: Unique detection ID within the frame.
        bbox: Bounding box [x1, y1, x2, y2].
        confidence: Detection confidence score.
    """

    def __init__(self, frame_id: int, detection_id: int,
                 bbox: np.ndarray, confidence: float = 1.0):
        self.frame_id = frame_id
        self.detection_id = detection_id
        self.bbox = bbox
        self.confidence = confidence

    @property
    def key(self) -> Tuple[int, int]:
        """Unique key for this node: (frame_id, detection_id)."""
        return (self.frame_id, self.detection_id)

    def __repr__(self) -> str:
        return f"TrackNode(frame={self.frame_id}, det={self.detection_id})"


class TrackGraph:
    """Graph of detections across frames for trajectory analysis.
    
    Uses adjacency list representation. Edges connect detections
    in nearby frames that have sufficient IoU overlap.
    
    Args:
        iou_threshold: Min IoU to create an edge between two detections.
        max_frame_gap: Max frame distance to consider for edges.
    """

    def __init__(self, iou_threshold: float = 0.2, max_frame_gap: int = 5):
        self.iou_threshold = iou_threshold
        self.max_frame_gap = max_frame_gap

        self.nodes: Dict[Tuple[int, int], TrackNode] = {}
        self.adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
        self.frame_detections: Dict[int, List[TrackNode]] = defaultdict(list)

    def add_detection(self, frame_id: int, detection_id: int,
                      bbox: np.ndarray, confidence: float = 1.0) -> None:
        """Add a detection node to the graph."""
        node = TrackNode(frame_id, detection_id, bbox, confidence)
        self.nodes[node.key] = node
        self.frame_detections[frame_id].append(node)

    def build_edges(self) -> None:
        """Build edges between detections in nearby frames based on IoU.
        
        For each pair of frames within max_frame_gap, creates bidirectional
        edges between detections with IoU >= iou_threshold.
        Weight = IoU * confidence of the target detection.
        """
        sorted_frames = sorted(self.frame_detections.keys())

        for i, frame_id in enumerate(sorted_frames):
            detections_current = self.frame_detections[frame_id]

            for j in range(i + 1, min(i + self.max_frame_gap + 1, len(sorted_frames))):
                next_frame = sorted_frames[j]
                detections_next = self.frame_detections[next_frame]

                for det_curr in detections_current:
                    for det_next in detections_next:
                        iou = compute_iou(det_curr.bbox, det_next.bbox)
                        
                        if iou >= self.iou_threshold:
                            weight = iou * det_next.confidence
                            self.adjacency[det_curr.key].append((det_next.key, weight))
                            self.adjacency[det_next.key].append((det_curr.key, weight))

    def find_track_path_dfs(self, start_key: Tuple[int, int],
                            forward_only: bool = True) -> List[TrackNode]:
        """Find the longest trajectory from a starting detection using DFS.
        
        Uses recursive DFS with backtracking, exploring highest-weight
        neighbors first (best-first DFS). Tracks the longest path found.
        
        Args:
            start_key: (frame_id, detection_id) starting point.
            forward_only: If True, only follow edges to later frames.
        
        Returns:
            List of TrackNodes forming the trajectory, sorted by frame.
        """
        if start_key not in self.nodes:
            return []

        visited: Set[Tuple[int, int]] = set()
        best_path: List[TrackNode] = []

        def _dfs(current_key: Tuple[int, int], current_path: List[TrackNode]):
            nonlocal best_path

            visited.add(current_key)
            current_node = self.nodes[current_key]
            current_path.append(current_node)

            # Update best path if current is longer
            if len(current_path) > len(best_path):
                best_path = current_path.copy()

            # Explore neighbors sorted by weight (best first)
            neighbors = sorted(
                self.adjacency.get(current_key, []),
                key=lambda x: x[1],
                reverse=True
            )

            for neighbor_key, weight in neighbors:
                if neighbor_key in visited:
                    continue
                if forward_only and neighbor_key[0] <= current_key[0]:
                    continue
                _dfs(neighbor_key, current_path)

            # Backtrack
            current_path.pop()
            visited.remove(current_key)

        _dfs(start_key, [])

        best_path.sort(key=lambda n: n.frame_id)
        return best_path

    def find_all_connected_tracks(self) -> List[List[TrackNode]]:
        """Find all connected components using iterative DFS.
        
        Each connected component represents a set of detections
        that could belong to the same object across frames.
        
        Returns:
            List of components, each a list of TrackNodes sorted by frame.
        """
        visited: Set[Tuple[int, int]] = set()
        components: List[List[TrackNode]] = []

        def _dfs_component(start_key: Tuple[int, int]) -> List[TrackNode]:
            """Iterative DFS using a stack to find one connected component."""
            stack = [start_key]
            component = []

            while stack:
                key = stack.pop()
                if key in visited:
                    continue
                
                visited.add(key)
                component.append(self.nodes[key])

                for neighbor_key, _ in self.adjacency.get(key, []):
                    if neighbor_key not in visited:
                        stack.append(neighbor_key)

            return component

        for node_key in self.nodes:
            if node_key not in visited:
                component = _dfs_component(node_key)
                if component:
                    component.sort(key=lambda n: n.frame_id)
                    components.append(component)

        return components
