"""
Track Associator — BFS-based Re-Identification
=================================================
Uses BFS (Breadth-First Search) to find the shortest path
between detections across frames. This enables reconnecting
tracks that were lost due to occlusion.

BFS guarantees shortest path (minimum frame gap), making it
ideal for finding the closest match to resume a lost track.
"""

from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ..training.metrics import compute_iou


class TrackAssociator:
    """Re-identification module using BFS for track repair.
    
    When a track is lost (e.g., due to occlusion), this module
    searches for the nearest matching detection to resume tracking.
    
    BFS explores frame-by-frame (level-order), guaranteeing
    the shortest path = minimum frame gap to the match.
    
    Args:
        max_frame_gap: Max frames to search for re-identification.
        iou_threshold: Min IoU for creating association edges.
        appearance_threshold: Optional appearance similarity threshold.
    """

    def __init__(self, max_frame_gap: int = 10,
                 iou_threshold: float = 0.15,
                 appearance_threshold: float = 0.3):
        self.max_frame_gap = max_frame_gap
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold

        # Detections per frame: frame_id -> [(det_id, bbox, features), ...]
        self.detections: Dict[int, List[Tuple[int, np.ndarray, Optional[np.ndarray]]]] = defaultdict(list)
        
        # Adjacency list for association graph
        self.adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    def add_frame_detections(self, frame_id: int,
                             det_ids: List[int],
                             bboxes: List[np.ndarray],
                             features: Optional[List[np.ndarray]] = None) -> None:
        """Register detections from a frame for later association.
        
        Args:
            frame_id: Frame number.
            det_ids: List of detection IDs.
            bboxes: List of bounding boxes.
            features: Optional feature vectors for appearance matching.
        """
        for i, (did, bbox) in enumerate(zip(det_ids, bboxes)):
            feat = features[i] if features else None
            self.detections[frame_id].append((did, bbox, feat))

    def build_association_graph(self) -> None:
        """Build adjacency graph between detections across nearby frames.
        
        Creates bidirectional edges between detections with IoU >= threshold.
        """
        sorted_frames = sorted(self.detections.keys())

        for i, frame_a in enumerate(sorted_frames):
            for j in range(i + 1, min(i + self.max_frame_gap + 1, len(sorted_frames))):
                frame_b = sorted_frames[j]
                
                for did_a, bbox_a, feat_a in self.detections[frame_a]:
                    for did_b, bbox_b, feat_b in self.detections[frame_b]:
                        iou = compute_iou(bbox_a, bbox_b)
                        
                        if iou >= self.iou_threshold:
                            key_a = (frame_a, did_a)
                            key_b = (frame_b, did_b)
                            self.adj[key_a].append(key_b)
                            self.adj[key_b].append(key_a)

    def find_shortest_association_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS to find the shortest path between two detections.
        
        BFS guarantees shortest path because it explores nodes
        level-by-level (FIFO queue), so the first time we reach
        the target is always via the minimum number of hops.
        
        Args:
            start: (frame_id, det_id) last known detection of lost track.
            end: (frame_id, det_id) candidate for re-identification.
        
        Returns:
            List of (frame_id, det_id) forming the shortest path, or None.
        """
        if start == end:
            return [start]

        queue: deque = deque()
        visited: Set[Tuple[int, int]] = set()
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

        queue.append(start)
        visited.add(start)
        parent[start] = None

        while queue:
            current = queue.popleft()  # FIFO: process closest nodes first

            for neighbor in self.adj.get(current, []):
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                parent[neighbor] = current

                if neighbor == end:
                    # Reconstruct path from end to start
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path

                queue.append(neighbor)

        return None  # No path found

    def find_reidentification_candidates(
        self,
        lost_track_key: Tuple[int, int],
        max_depth: int = 5
    ) -> List[Tuple[Tuple[int, int], int]]:
        """BFS to find all reachable detections within max_depth hops.
        
        Returns candidates ordered by distance (closest first),
        which BFS guarantees naturally.
        
        Args:
            lost_track_key: (frame_id, det_id) of the lost track.
            max_depth: Max BFS depth to search.
        
        Returns:
            List of ((frame_id, det_id), distance) tuples.
        """
        candidates: List[Tuple[Tuple[int, int], int]] = []
        visited: Set[Tuple[int, int]] = set()
        queue: deque = deque()

        queue.append((lost_track_key, 0))
        visited.add(lost_track_key)

        while queue:
            current, depth = queue.popleft()

            if depth > 0:  # Don't count the starting node
                candidates.append((current, depth))

            if depth >= max_depth:
                continue

            for neighbor in self.adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return candidates

    def repair_track(self, lost_track_last: Tuple[int, int],
                     new_detections: List[Tuple[int, int]]
                     ) -> Optional[Tuple[int, int]]:
        """Find the best re-identification match for a lost track.
        
        Uses BFS to find the closest candidate among new detections.
        
        Args:
            lost_track_last: Last detection of the lost track.
            new_detections: List of new detection keys to consider.
        
        Returns:
            Best matching detection key, or None.
        """
        candidates = self.find_reidentification_candidates(
            lost_track_last, max_depth=self.max_frame_gap
        )

        new_det_set = set(new_detections)
        
        # Return the closest candidate that's in new_detections
        for candidate_key, distance in candidates:
            if candidate_key in new_det_set:
                return candidate_key

        return None
