"""
Hierarchical NMS — BFS-based Non-Maximum Suppression
=======================================================
Groups overlapping detections into clusters using BFS
(connected components), then keeps the highest-scoring
detection from each cluster.

Compared to greedy NMS: order-independent and more flexible.
"""

from collections import deque
from typing import Dict, List, Set, Tuple
import numpy as np

from ..training.metrics import compute_iou


def hierarchical_nms(boxes: np.ndarray,
                     scores: np.ndarray,
                     iou_threshold: float = 0.5) -> List[int]:
    """BFS-based Hierarchical NMS.
    
    Algorithm:
        1. Build graph: detections as nodes, edges where IoU >= threshold
        2. BFS to find connected components (overlapping clusters)
        3. From each component: keep the highest-scoring detection
    
    Args:
        boxes: (N, 4) array of bboxes [x1, y1, x2, y2].
        scores: (N,) array of confidence scores.
        iou_threshold: IoU threshold for considering overlap.
    
    Returns:
        List of indices of boxes kept after NMS.
    """
    if len(boxes) == 0:
        return []

    n = len(boxes)

    # Step 1: Build adjacency graph
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(boxes[i], boxes[j])
            if iou >= iou_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Step 2: BFS to find connected components
    visited: Set[int] = set()
    components: List[List[int]] = []

    for start_node in range(n):
        if start_node in visited:
            continue

        component: List[int] = []
        queue: deque = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    # Step 3: Keep highest-scoring detection from each component
    keep_indices: List[int] = []
    
    for component in components:
        best_idx = max(component, key=lambda i: scores[i])
        keep_indices.append(best_idx)

    keep_indices.sort(key=lambda i: scores[i], reverse=True)

    return keep_indices
