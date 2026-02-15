"""
Model Evaluator — Computes mAP@0.5 on Validation Set
=======================================================
Runs inference on the validation set, matches predictions to
ground truth using IoU, and computes per-class AP and mAP.
"""

from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_iou, compute_precision_recall, compute_ap


class Evaluator:
    """Evaluator for object detection — computes mAP@0.5.
    
    Pipeline: inference -> collect predictions/GT -> compute precision-recall -> compute AP
    
    Args:
        device: Computing device ('cpu' or 'cuda').
        iou_threshold: IoU threshold for matching (default 0.5, Pascal VOC standard).
    """

    def __init__(self, device: str = 'cpu', iou_threshold: float = 0.5):
        self.device = device
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module,
                 data_loader: DataLoader,
                 num_classes: int = 2) -> Dict[str, float]:
        """Evaluate model on validation set and compute mAP@0.5.
        
        Args:
            model: PyTorch model (FasterRCNN).
            data_loader: Validation DataLoader.
            num_classes: Number of classes (including background).
        
        Returns:
            Dict with mAP@0.5 and per-class AP values.
        """
        model.eval()

        # Collect predictions and ground truths per class
        all_pred_boxes: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, num_classes)}
        all_pred_scores: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}
        all_gt_boxes: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, num_classes)}

        for images, targets in data_loader:
            images = [img.to(self.device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Collect predictions
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if 1 <= label < num_classes:
                        all_pred_boxes[label].append(box)
                        all_pred_scores[label].append(score)

                # Collect ground truths
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()

                for box, label in zip(gt_boxes, gt_labels):
                    if 1 <= label < num_classes:
                        all_gt_boxes[label].append(box)

        # Compute AP for each class
        aps = {}

        for class_id in range(1, num_classes):
            pred_b = all_pred_boxes[class_id]
            pred_s = all_pred_scores[class_id]
            gt_b = all_gt_boxes[class_id]

            if len(gt_b) == 0:
                aps[class_id] = 0.0
                continue

            precision, recall = compute_precision_recall(
                pred_b, pred_s, gt_b, self.iou_threshold
            )

            ap = compute_ap(precision, recall)
            aps[class_id] = ap

        # Compute mean AP across all classes
        if aps:
            mAP = np.mean(list(aps.values()))
        else:
            mAP = 0.0

        results = {'mAP@0.5': float(mAP)}
        for class_id, ap in aps.items():
            results[f'AP_class_{class_id}'] = float(ap)

        return results
