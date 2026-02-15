"""
COCO to YOLO Dataset Converter
==================================
Converts COCO-format annotations (from Roboflow) to YOLO txt format
required by Ultralytics for training.

YOLO format per line:  class_id x_center y_center width height
All values are normalized to [0, 1] relative to image dimensions.

Also generates the data.yaml config file for Ultralytics training.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def convert_coco_to_yolo(coco_dir: str, output_dir: str,
                          splits: Optional[List[str]] = None) -> str:
    """Convert COCO dataset to YOLO format.

    Reads _annotations.coco.json from each split directory and
    creates corresponding YOLO .txt label files.

    Args:
        coco_dir: Root directory with COCO-format data (train/valid/test).
        output_dir: Output directory for YOLO-format dataset.
        splits: List of splits to convert ('train', 'valid', 'test').

    Returns:
        Path to generated data.yaml file.
    """
    if splits is None:
        splits = ['train', 'valid', 'test']

    output_path = Path(output_dir)
    class_names: List[str] = []
    cat_id_to_idx: Dict[int, int] = {}

    for split in splits:
        split_dir = Path(coco_dir) / split
        ann_file = split_dir / '_annotations.coco.json'

        if not ann_file.exists():
            print(f"  ⚠ Skipping {split}: {ann_file} not found")
            continue

        # Load COCO annotations
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Build category mapping (skip dummy 'objects' category)
        if not cat_id_to_idx:
            idx = 0
            for cat in coco_data['categories']:
                if cat.get('supercategory') == 'none' and cat['name'] == 'objects':
                    continue
                cat_id_to_idx[cat['id']] = idx
                class_names.append(cat['name'])
                idx += 1

        # Build image_id → image_info lookup
        id_to_info = {img['id']: img for img in coco_data['images']}

        # Build image_id → annotations lookup
        id_to_anns: Dict[int, list] = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in id_to_anns:
                id_to_anns[img_id] = []
            id_to_anns[img_id].append(ann)

        # Create output directories
        img_out = output_path / 'images' / split
        lbl_out = output_path / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        converted = 0

        for img_info in coco_data['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            # Copy image to output
            src_img = split_dir / file_name
            dst_img = img_out / file_name

            if src_img.exists() and not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))

            # Convert annotations to YOLO format
            anns = id_to_anns.get(img_id, [])
            label_file = lbl_out / (Path(file_name).stem + '.txt')

            with open(label_file, 'w') as f:
                for ann in anns:
                    cat_id = ann['category_id']
                    if cat_id not in cat_id_to_idx:
                        continue

                    class_idx = cat_id_to_idx[cat_id]
                    x, y, w, h = ann['bbox']  # COCO: [x, y, width, height]

                    # Convert to YOLO: [x_center, y_center, width, height] normalized
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h

                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))

                    # Skip degenerate boxes
                    if w_norm <= 0 or h_norm <= 0:
                        continue

                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} "
                            f"{w_norm:.6f} {h_norm:.6f}\n")

            converted += 1

        print(f"  ✅ {split}: {converted} images converted")

    # Generate data.yaml
    data_yaml_path = output_path / 'data.yaml'
    data_config = {
        'path': str(output_path.resolve()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names,
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"\n📄 data.yaml generated: {data_yaml_path}")
    print(f"   Classes ({len(class_names)}): {class_names}")

    return str(data_yaml_path)


def prepare_yolo_dataset(coco_dir: str = 'data/coco',
                          output_dir: str = 'data/yolo') -> str:
    """High-level function to prepare YOLO dataset from COCO export.

    Args:
        coco_dir: Path to COCO-format dataset root.
        output_dir: Path for YOLO-format output.

    Returns:
        Path to data.yaml file.
    """
    print("=" * 60)
    print("  Converting COCO → YOLO Format")
    print("=" * 60)
    print(f"  Input:  {coco_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    data_yaml = convert_coco_to_yolo(coco_dir, output_dir)

    # Print dataset statistics
    output_path = Path(output_dir)
    for split in ['train', 'valid', 'test']:
        img_dir = output_path / 'images' / split
        lbl_dir = output_path / 'labels' / split
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob('*.[jp][pn][g]')))
            n_lbls = len(list(lbl_dir.glob('*.txt')))
            print(f"  {split:6s}: {n_imgs:5d} images, {n_lbls:5d} labels")

    print("=" * 60)

    return data_yaml


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format')
    parser.add_argument('--input', default='data/coco', help='COCO dataset root')
    parser.add_argument('--output', default='data/yolo', help='YOLO output dir')
    args = parser.parse_args()

    prepare_yolo_dataset(args.input, args.output)
