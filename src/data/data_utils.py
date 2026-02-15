"""
Data Utilities
================
Custom collate function for DataLoader and visualization helpers.
"""

from typing import Dict, List, Tuple

import torch
import numpy as np
import cv2
from PIL import Image


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
               ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Custom collate function for object detection DataLoader.
    
    Since each image can have a different number of bounding boxes,
    we can't stack targets into a single tensor. Instead, we return
    lists of images and targets, which FasterRCNN accepts directly.
    
    Args:
        batch: List of (image, target) tuples from Dataset.__getitem__.
    
    Returns:
        Tuple of (image_list, target_list).
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def visualize_sample(image: torch.Tensor,
                     target: Dict[str, torch.Tensor],
                     class_names: List[str] = None,
                     save_path: str = None) -> np.ndarray:
    """Draw bounding boxes on image for visual debugging.
    
    Useful for verifying that annotations and transforms are correct.
    
    Args:
        image: Image tensor (3, H, W), values in [0, 1].
        target: Target dict with 'boxes' and 'labels'.
        class_names: Optional list of class name strings.
        save_path: Optional path to save the output image.
    
    Returns:
        BGR numpy array (H, W, 3) with drawn bounding boxes.
    """
    # Convert tensor to numpy BGR image for OpenCV
    img = image.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label text
        if class_names and label > 0:
            name = class_names[label - 1] if label <= len(class_names) else f'cls_{label}'
        else:
            name = f'cls_{label}'

        cv2.putText(img, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

    if save_path:
        cv2.imwrite(save_path, img)

    return img
