"""
Data Augmentation Transforms for Object Detection
===================================================
Custom transform classes that handle both image transformations
and bounding box coordinate updates.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance


class Compose:
    """Chain multiple transforms together into a pipeline.
    
    Args:
        transforms: List of transform callables.
    """

    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """Apply each transform sequentially."""
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """Resize image and scale bounding boxes accordingly.
    
    Maintains aspect ratio by scaling to min_size on the shorter side,
    while ensuring the longer side doesn't exceed max_size.
    
    Args:
        min_size: Target size for the shorter side.
        max_size: Maximum allowed size for the longer side.
    """

    def __init__(self, min_size: int = 640, max_size: int = 800):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        w, h = image.size

        # Calculate scale factor to resize shorter side to min_size
        scale = self.min_size / min(h, w)
        
        # If longer side exceeds max_size, use smaller scale
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)

        # Compute new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image using bilinear interpolation
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Scale bounding boxes by the same factors
        if len(target['boxes']) > 0:
            sx = new_w / w  # Width scale factor
            sy = new_h / h  # Height scale factor
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= sx  # Scale x coordinates
            boxes[:, [1, 3]] *= sy  # Scale y coordinates
            target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and boxes horizontally.
    
    Args:
        prob: Probability of flipping (default 0.5).
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if len(target['boxes']) > 0:
                w = image.size[0]
                boxes = target['boxes'].clone()
                # Mirror x coordinates: x' = W - x
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                target['boxes'] = boxes

        return image, target


class ColorJitter:
    """Random color jitter: brightness, contrast, saturation.
    
    Does not affect bounding boxes (only modifies pixel values).
    
    Args:
        brightness: Max brightness change.
        contrast: Max contrast change.
        saturation: Max saturation change.
    """

    def __init__(self, brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        return image, target


class GaussianBlur:
    """Apply Gaussian blur randomly for regularization.
    
    Args:
        prob: Probability of applying blur.
        radius_range: Range of blur radius.
    """

    def __init__(self, prob: float = 0.3,
                 radius_range: Tuple[float, float] = (0.5, 1.5)):
        self.prob = prob
        self.radius_range = radius_range

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image, target


class ToTensor:
    """Convert PIL Image to PyTorch tensor.
    
    Normalizes pixel values from [0, 255] to [0, 1] and
    transposes from (H, W, C) to (C, H, W) format.
    """

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_array = np.array(image, dtype=np.float32)
        img_array /= 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor, target


class TrainTransform:
    """Training augmentation pipeline.
    
    Applies: Resize → Horizontal Flip → Color Jitter → Blur → ToTensor
    """

    def __init__(self, min_size: int = 640, max_size: int = 800):
        self.pipeline = Compose([
            Resize(min_size, max_size),
            RandomHorizontalFlip(prob=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            GaussianBlur(prob=0.3),
            ToTensor(),
        ])

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.pipeline(image, target)


class ValTransform:
    """Validation/test transform pipeline (no augmentation).
    
    Applies: Resize → ToTensor only
    """

    def __init__(self, min_size: int = 640, max_size: int = 800):
        self.pipeline = Compose([
            Resize(min_size, max_size),
            ToTensor(),
        ])

    def __call__(self, image: Image.Image,
                 target: Dict[str, torch.Tensor]
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.pipeline(image, target)
