"""
Dataset Download Script — Roboflow Integration
================================================
Download license plate dataset from Roboflow workspace.
Supports both COCO and YOLOv8 formats.

Handles Roboflow's output directory structure automatically:
Roboflow downloads to: <output>/<ProjectName-Version>/train/, valid/, test/
This script moves files to: data/coco/train/, valid/, test/
"""

import argparse
import os
import sys
import shutil
import glob
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed.")
    print("Install it with: pip install roboflow")
    sys.exit(1)


def find_dataset_splits(search_dir: str):
    """Find train/valid/test directories recursively under search_dir.
    
    Roboflow downloads to unpredictable subdirectory names like
    'My-First-Project-2/', so we search for the split folders.
    
    Returns:
        dict with 'train', 'valid', 'test' paths, or None if not found.
    """
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        # Search for split directory
        for root, dirs, files in os.walk(search_dir):
            if os.path.basename(root) == split_name:
                # Check if this directory has images or annotation files
                has_content = any(
                    f.endswith(('.jpg', '.jpeg', '.png', '.json'))
                    for f in files
                )
                if has_content:
                    splits[split_name] = root
                    break
    return splits if splits else None


def count_images(directory: str) -> int:
    """Count image files in a directory."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        count += len(glob.glob(os.path.join(directory, ext)))
    return count


def download_dataset(api_key: str, 
                     workspace: str,
                     project: str,
                     version: int,
                     fmt: str = "coco",
                     output_dir: str = "data") -> None:
    """
    Download dataset from Roboflow and organize into data/coco/ structure.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version number
        fmt: Dataset format ("coco" or "yolov8")
        output_dir: Output directory path
    """
    print(f"\n{'='*60}")
    print(f"Downloading Dataset from Roboflow")
    print(f"{'='*60}")
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    print(f"Format: {fmt}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Target directory: data/coco/
    target_dir = os.path.join(output_dir, "coco")
    
    try:
        # Initialize Roboflow
        print("loading Roboflow workspace...")
        rf = Roboflow(api_key=api_key)
        
        print("loading Roboflow project...")
        project_obj = rf.workspace(workspace).project(project)
        version_obj = project_obj.version(version)
        
        # Download to a temp location first
        temp_download = os.path.join(output_dir, "_roboflow_download")
        os.makedirs(temp_download, exist_ok=True)
        
        print("Downloading... (this may take a few minutes)")
        dataset = version_obj.download(fmt, location=temp_download)
        
        download_location = dataset.location if hasattr(dataset, 'location') else temp_download
        print(f"Roboflow downloaded to: {download_location}")
        
        # Find the actual split directories
        splits = find_dataset_splits(download_location)
        
        if not splits:
            # Also check temp_download root
            splits = find_dataset_splits(temp_download)
        
        if not splits:
            # Last resort: check output_dir
            splits = find_dataset_splits(output_dir)
        
        if not splits:
            print("\n❌ Could not find train/valid/test directories!")
            print(f"Please check the contents of: {download_location}")
            print("Files found:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:
                    print(f'{subindent}{file}')
                if len(files) > 5:
                    print(f'{subindent}... and {len(files)-5} more files')
            sys.exit(1)
        
        # Move splits to target directory
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\nOrganizing dataset into {target_dir}/")
        for split_name, split_path in splits.items():
            dest = os.path.join(target_dir, split_name)
            if os.path.exists(dest) and dest != split_path:
                shutil.rmtree(dest)
            
            if split_path != dest:
                shutil.move(split_path, dest)
                print(f"  ✓ Moved {split_name}/ → {dest}")
            else:
                print(f"  ✓ {split_name}/ already in place")
        
        # Clean up temp download directory
        if os.path.exists(temp_download):
            shutil.rmtree(temp_download, ignore_errors=True)
        
        # Also clean any leftover Roboflow directories
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item not in ['coco', 'yolov8']:
                shutil.rmtree(item_path, ignore_errors=True)
        
        # Count and report
        print(f"\n{'='*60}")
        print(f"✅ Download Complete!")
        print(f"{'='*60}")
        print(f"Location: {os.path.abspath(target_dir)}")
        print(f"Format: {fmt}")
        
        train_count = count_images(os.path.join(target_dir, "train"))
        valid_count = count_images(os.path.join(target_dir, "valid"))
        test_count = count_images(os.path.join(target_dir, "test"))
        total = train_count + valid_count + test_count
        
        print(f"\nDataset Statistics:")
        print(f"  Train: {train_count} images")
        print(f"  Valid: {valid_count} images")
        print(f"  Test:  {test_count} images")
        print(f"  Total: {total} images")
        
        # Verify annotations
        for split in ['train', 'valid', 'test']:
            ann_file = os.path.join(target_dir, split, '_annotations.coco.json')
            if os.path.exists(ann_file):
                size_mb = os.path.getsize(ann_file) / (1024 * 1024)
                print(f"  ✓ {split}/_annotations.coco.json ({size_mb:.1f} MB)")
            else:
                print(f"  ⚠ {split}/_annotations.coco.json NOT FOUND")
        
        print(f"{'='*60}\n")
        
        if total == 0:
            print("⚠ WARNING: 0 images found! Check directory structure:")
            print(f"  ls -la {target_dir}/train/")
            print(f"  ls -la {target_dir}/valid/")
            print(f"  ls -la {target_dir}/test/")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download license plate dataset from Roboflow"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default="EmY26Ik4IZIW8iOyGxVL",
        help="Roboflow API key (default: project key)"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="nguyn-cng-tuyn",
        help="Roboflow workspace name"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="my-first-project-usuhh-7gecz",
        help="Roboflow project name"
    )
    
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Dataset version number"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "yolov8"],
        default="coco",
        help="Dataset format (coco or yolov8)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory path"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Download dataset
    download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        fmt=args.format,
        output_dir=args.output
    )
    
    print("Ready to train! Run:")
    print("  python -m src train --config configs/train_plate_detector.yaml --device cuda")


if __name__ == "__main__":
    main()
