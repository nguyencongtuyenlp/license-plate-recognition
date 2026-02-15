"""
Advanced Data Augmentation for Object Detection
==================================================
Implements training-time augmentation techniques commonly used
in YOLO and modern object detectors.

Techniques:
    - Mosaic: Combines 4 images into a single training sample
    - MixUp: Alpha-blends two images for regularization
    - CutOut: Random rectangular erasing for occlusion robustness

References:
    - Mosaic: Bochkovskiy et al., 'YOLOv4', arXiv:2004.10934
    - MixUp: Zhang et al., 'mixup: Beyond Empirical Risk Minimization'
    - CutOut: DeVries & Taylor, 'Improved Regularization of CNNs'
"""

import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image


class MosaicAugmentation:
    """Mosaic augmentation — combines 4 images into one.

    Creates a 2x2 grid of randomly selected images, with a random
    split point. This dramatically increases the effective batch size
    and forces the model to detect objects at various positions.

    Benefits:
        - 4x more context per training sample
        - Objects appear at different scales and positions
        - Reduces need for large batch sizes
        - Key innovation in YOLOv4/v5/v8

    Args:
        img_size: Target output image size.
        dataset: Reference to the dataset for sampling other images.
    """

    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def __call__(self, images: List[np.ndarray],
                 targets: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict]:
        """Apply mosaic augmentation on 4 images.

        Args:
            images: List of 4 numpy images (H, W, 3), RGB.
            targets: List of 4 target dicts with 'boxes' and 'labels'.

        Returns:
            Tuple of (mosaic_image, merged_target).
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"

        s = self.img_size

        # Random center point for the mosaic grid
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))

        # Output canvas
        mosaic_img = np.zeros((s, s, 3), dtype=np.uint8)

        all_boxes = []
        all_labels = []

        # Placement regions for each of the 4 images
        # (x1_place, y1_place, x2_place, y2_place) in output canvas
        placements = [
            (0, 0, xc, yc),                    # top-left
            (xc, 0, s, yc),                     # top-right
            (0, yc, xc, s),                     # bottom-left
            (xc, yc, s, s),                     # bottom-right
        ]

        for i, (img, target) in enumerate(zip(images, targets)):
            h, w = img.shape[:2]
            x1p, y1p, x2p, y2p = placements[i]
            pw, ph = x2p - x1p, y2p - y1p

            # Compute crop region from source image
            # Scale source to fit the placement region
            scale = min(pw / w, ph / h)
            new_w, new_h = int(w * scale), int(h * scale)

            if new_w <= 0 or new_h <= 0:
                continue

            # Resize source image
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
            resized = np.array(pil_img)

            # Compute offset to center in placement region
            dx = (pw - new_w) // 2
            dy = (ph - new_h) // 2

            # Place resized image on canvas
            cx1 = x1p + dx
            cy1 = y1p + dy
            cx2 = cx1 + new_w
            cy2 = cy1 + new_h

            # Clip to canvas bounds
            cx1_c, cy1_c = max(cx1, 0), max(cy1, 0)
            cx2_c, cy2_c = min(cx2, s), min(cy2, s)

            # Corresponding region in resized image
            sx1 = cx1_c - cx1
            sy1 = cy1_c - cy1
            sx2 = sx1 + (cx2_c - cx1_c)
            sy2 = sy1 + (cy2_c - cy1_c)

            if cx2_c > cx1_c and cy2_c > cy1_c:
                mosaic_img[cy1_c:cy2_c, cx1_c:cx2_c] = resized[sy1:sy2, sx1:sx2]

            # Transform bounding boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone() if isinstance(target['boxes'], torch.Tensor) \
                    else torch.tensor(target['boxes'], dtype=torch.float32)
                labels = target['labels'].clone() if isinstance(target['labels'], torch.Tensor) \
                    else torch.tensor(target['labels'], dtype=torch.int64)

                # Scale boxes to resized dimensions
                boxes[:, [0, 2]] *= scale
                boxes[:, [1, 3]] *= scale

                # Offset to canvas position
                boxes[:, [0, 2]] += cx1
                boxes[:, [1, 3]] += cy1

                # Clip to canvas
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, s)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, s)

                # Filter out degenerate boxes
                w_box = boxes[:, 2] - boxes[:, 0]
                h_box = boxes[:, 3] - boxes[:, 1]
                valid = (w_box > 2) & (h_box > 2)

                if valid.any():
                    all_boxes.append(boxes[valid])
                    all_labels.append(labels[valid])

        # Merge all boxes and labels
        if all_boxes:
            merged_boxes = torch.cat(all_boxes, dim=0)
            merged_labels = torch.cat(all_labels, dim=0)
        else:
            merged_boxes = torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = torch.zeros((0,), dtype=torch.int64)

        merged_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'area': (merged_boxes[:, 2] - merged_boxes[:, 0]) *
                    (merged_boxes[:, 3] - merged_boxes[:, 1]),
            'iscrowd': torch.zeros(len(merged_boxes), dtype=torch.int64),
        }

        return mosaic_img, merged_target


class MixUpAugmentation:
    """MixUp augmentation — alpha-blends two images.

    Creates a convex combination of two training images and their
    labels, acting as a strong regularizer.

    Formula:
        image_mix = α * image_1 + (1 - α) * image_2
        target_mix = merge(target_1, target_2)

    where α ~ Beta(beta, beta), typically beta = 1.5.

    Args:
        alpha: Beta distribution parameter (default 1.5).
    """

    def __init__(self, alpha: float = 1.5):
        self.alpha = alpha

    def __call__(self, image1: np.ndarray, target1: Dict,
                 image2: np.ndarray, target2: Dict) -> Tuple[np.ndarray, Dict]:
        """Apply MixUp to two image-target pairs.

        Args:
            image1, target1: First image and annotations.
            image2, target2: Second image and annotations.

        Returns:
            Mixed image and merged annotations.
        """
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure primary image dominates

        # Resize image2 to match image1
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if (h1, w1) != (h2, w2):
            pil_img2 = Image.fromarray(image2)
            pil_img2 = pil_img2.resize((w1, h1), Image.BILINEAR)
            image2 = np.array(pil_img2)

            # Scale boxes for image2
            if 'boxes' in target2 and len(target2['boxes']) > 0:
                sx = w1 / w2
                sy = h1 / h2
                boxes2 = target2['boxes'].clone() if isinstance(target2['boxes'], torch.Tensor) \
                    else torch.tensor(target2['boxes'], dtype=torch.float32)
                boxes2[:, [0, 2]] *= sx
                boxes2[:, [1, 3]] *= sy
                target2 = {**target2, 'boxes': boxes2}

        # Blend images
        mixed = (lam * image1.astype(np.float32) +
                 (1 - lam) * image2.astype(np.float32)).astype(np.uint8)

        # Merge targets (keep all boxes from both images)
        boxes1 = target1.get('boxes', torch.zeros((0, 4)))
        labels1 = target1.get('labels', torch.zeros((0,), dtype=torch.int64))
        boxes2 = target2.get('boxes', torch.zeros((0, 4)))
        labels2 = target2.get('labels', torch.zeros((0,), dtype=torch.int64))

        if isinstance(boxes1, torch.Tensor) and isinstance(boxes2, torch.Tensor):
            merged_boxes = torch.cat([boxes1, boxes2], dim=0)
            merged_labels = torch.cat([labels1, labels2], dim=0)
        else:
            merged_boxes = torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = torch.zeros((0,), dtype=torch.int64)

        merged_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'area': (merged_boxes[:, 2] - merged_boxes[:, 0]) *
                    (merged_boxes[:, 3] - merged_boxes[:, 1]),
            'iscrowd': torch.zeros(len(merged_boxes), dtype=torch.int64),
        }

        return mixed, merged_target


class CutOutAugmentation:
    """CutOut augmentation — randomly erases rectangular patches.

    Forces the model to focus on multiple visual cues rather than
    relying on a single discriminative region. This improves robustness
    to partial occlusion (common with license plates).

    Args:
        num_holes: Number of rectangular patches to erase.
        max_h_size: Maximum height of erased patch.
        max_w_size: Maximum width of erased patch.
        fill_value: Pixel value to fill erased regions (default 114, YOLO gray).
    """

    def __init__(self, num_holes: int = 3, max_h_size: int = 64,
                 max_w_size: int = 64, fill_value: int = 114):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def __call__(self, image: np.ndarray,
                 target: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply CutOut augmentation.

        Args:
            image: Input image (H, W, 3).
            target: Optional target dict (passed through unchanged).

        Returns:
            Image with random patches erased, and unchanged target.
        """
        h, w = image.shape[:2]
        result = image.copy()

        for _ in range(self.num_holes):
            # Random patch size
            patch_h = random.randint(1, self.max_h_size)
            patch_w = random.randint(1, self.max_w_size)

            # Random position
            cy = random.randint(0, h)
            cx = random.randint(0, w)

            y1 = max(0, cy - patch_h // 2)
            y2 = min(h, cy + patch_h // 2)
            x1 = max(0, cx - patch_w // 2)
            x2 = min(w, cx + patch_w // 2)

            result[y1:y2, x1:x2] = self.fill_value

        return result, target
