"""
COCO Dataset Loader for License Plate Detection
==================================================
Parses COCO JSON annotations from Roboflow and returns
(image, target) pairs for FasterRCNN training.
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class COCOPlateDataset(Dataset):
    """Dataset for license plate detection, loading from COCO JSON format.
    
    Converts COCO bbox format [x, y, w, h] to torchvision format [x1, y1, x2, y2].
    
    Args:
        root_dir: Root directory containing split folders (train/valid/test).
        split: One of 'train', 'valid', 'test'.
        transforms: Callable transform(image, target) -> (image, target).
    
    Example:
        >>> dataset = COCOPlateDataset("data/coco", split="train")
        >>> image, target = dataset[0]
        >>> print(image.shape)          # (3, H, W)
        >>> print(target['boxes'])      # tensor([[x1, y1, x2, y2], ...])
    """

    def __init__(self, root_dir: str, split: str = 'train',
                 transforms: Optional[Callable] = None):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, split)

        # Load COCO annotation file (Roboflow export format)
        ann_file = os.path.join(self.image_dir, '_annotations.coco.json')
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Expected _annotations.coco.json from Roboflow COCO export"
            )

        with open(ann_file, 'r', encoding='utf-8') as f:
            self.coco_data: Dict[str, Any] = json.load(f)

        self.images: List[Dict] = self.coco_data['images']
        self.annotations: List[Dict] = self.coco_data['annotations']
        self.categories: List[Dict] = self.coco_data['categories']

        # Build lookup index: image_id -> list of annotations (O(1) access)
        self._img_id_to_anns: Dict[int, List[Dict]] = self._build_annotation_index()

        # Remap category IDs to contiguous range [1, N]
        # Skips Roboflow's dummy 'objects' supercategory
        self._cat_remap: Dict[int, int] = self._build_category_remap()

        print(f"[COCOPlateDataset] Loaded {split}: "
              f"{len(self.images)} images, {len(self.annotations)} annotations, "
              f"{len(self.categories)} categories")

    def _build_annotation_index(self) -> Dict[int, List[Dict]]:
        """Build hash map from image_id to annotation list for O(1) lookup."""
        index: Dict[int, List[Dict]] = {}
        
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in index:
                index[img_id] = []
            index[img_id].append(ann)
        
        return index

    def _build_category_remap(self) -> Dict[int, int]:
        """Map COCO category IDs to contiguous range [1, N].
        
        torchvision requires: 0 = background, 1..N = object classes.
        Skips Roboflow's dummy 'objects' supercategory.
        
        Returns:
            Dict mapping original_cat_id -> contiguous_id.
        """
        remap = {}
        real_id = 1  # Start from 1 (0 reserved for background)
        
        for cat in self.categories:
            # Skip Roboflow dummy category
            if cat.get('supercategory') == 'none' and cat['name'] == 'objects':
                continue
            
            remap[cat['id']] = real_id
            real_id += 1
        
        return remap

    def __len__(self) -> int:
        """Return total number of images in this split."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Load image and annotations at given index.
        
        Pipeline: Load image -> Parse annotations -> Convert bbox format ->
                  Apply transforms -> Return (image_tensor, target_dict)
        
        Args:
            idx: Index into the images list.
        
        Returns:
            Tuple of (image_tensor, target_dict) where target contains:
                - 'boxes': FloatTensor (N, 4) in [x1, y1, x2, y2] format
                - 'labels': Int64Tensor (N,) class indices
                - 'image_id': Int64Tensor (1,)
                - 'area': FloatTensor (N,) bbox areas
                - 'iscrowd': Int64Tensor (N,) crowd flags
        """
        # Get image metadata
        img_info = self.images[idx]
        img_id = img_info['id']
        file_name = img_info['file_name']

        # Load image
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert('RGB')

        # Get annotations for this image
        anns = self._img_id_to_anns.get(img_id, [])

        # Parse bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: [x, y, width, height]

            # Convert to torchvision format: [x1, y1, x2, y2]
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self._cat_remap.get(ann['category_id'], 1))
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

        # Convert to tensors
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Negative sample (no objects) — still need proper tensor shapes
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'area': areas_tensor,
            'iscrowd': iscrowd_tensor,
        }

        # Apply transforms (augmentation pipeline)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = self._pil_to_tensor(image)

        return image, target

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to float tensor [0, 1], shape (C, H, W)."""
        img_array = np.array(image, dtype=np.float32)
        img_array /= 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor

    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get image metadata at index (without loading pixels)."""
        return self.images[idx]

    @property
    def num_classes(self) -> int:
        """Number of object classes (excluding background)."""
        return len(self.categories)

    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return [cat['name'] for cat in self.categories]
