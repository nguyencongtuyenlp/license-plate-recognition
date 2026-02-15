"""
Detection Metrics — IoU, Precision-Recall, and Average Precision
==================================================================
Hand-implemented metrics for evaluating object detection models.
No external metric libraries used.
"""

from typing import List, Tuple
import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes.
    
    IoU = intersection_area / union_area, where
    union = area_a + area_b - intersection (inclusion-exclusion principle).
    
    Args:
        box_a: First bbox [x1, y1, x2, y2].
        box_b: Second bbox [x1, y1, x2, y2].
    
    Returns:
        IoU value in [0, 1].
    """
    # Intersection coordinates
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Intersection area (clamp to 0 if no overlap)
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Individual box areas
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union area (inclusion-exclusion)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_precision_recall(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision and recall at all confidence thresholds.
    
    Algorithm:
        1. Sort predictions by confidence (descending)
        2. For each prediction, find best matching GT (by IoU)
        3. If IoU >= threshold and GT not already matched -> TP, else FP
        4. Compute cumulative precision and recall
    
    Args:
        pred_boxes: List of predicted bboxes [x1, y1, x2, y2].
        pred_scores: Corresponding confidence scores.
        gt_boxes: List of ground truth bboxes.
        iou_threshold: IoU threshold for a valid match (default 0.5).
    
    Returns:
        Tuple of (precision_array, recall_array).
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.array([]), np.array([])

    # Sort predictions by confidence (highest first)
    sorted_indices = np.argsort(-np.array(pred_scores))

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = set()  # Track which GT boxes have been matched

    total_gt = len(gt_boxes)

    for det_idx_pos, det_idx in enumerate(sorted_indices):
        pred_box = pred_boxes[det_idx]
        
        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching GT box
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Classify as TP or FP
        if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
            tp[det_idx_pos] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[det_idx_pos] = 1

    # Cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (total_gt + 1e-6)

    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation (Pascal VOC method).
    
    Samples precision at 11 recall points [0.0, 0.1, ..., 1.0]
    and takes the mean of max precision at each recall level.
    
    Args:
        precision: Precision array from compute_precision_recall.
        recall: Corresponding recall array.
    
    Returns:
        AP value in [0, 1].
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # 11 evenly-spaced recall points
    recall_points = np.linspace(0, 1, 11)

    ap = 0.0

    for r in recall_points:
        # Get max precision at recall >= r (interpolation)
        prec_at_recall = precision[recall >= r]

        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()

    ap /= 11.0
    return ap
