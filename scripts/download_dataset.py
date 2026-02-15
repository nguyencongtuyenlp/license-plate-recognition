"""
Dataset Download Script - Roboflow Integration
================================================
Download license plate dataset from Roboflow workspace.
Supports both COCO and YOLOv8 formats.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed.")
    print("Install it with: pip install roboflow")
    sys.exit(1)


def download_dataset(api_key: str, 
                     workspace: str,
                     project: str,
                     version: int,
                     format: str = "coco",
                     output_dir: str = "data") -> None:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version number
        format: Dataset format ("coco" or "yolov8")
        output_dir: Output directory path
    """
    print(f"\n{'='*60}")
    print(f"Downloading Dataset from Roboflow")
    print(f"{'='*60}")
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    print(f"Format: {format}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Get project
        project_obj = rf.workspace(workspace).project(project)
        
        # Get version
        version_obj = project_obj.version(version)
        
        # Download dataset
        print("Downloading... (this may take a few minutes)")
        dataset = version_obj.download(format, location=output_dir)
        
        print(f"\n{'='*60}")
        print(f"✓ Download Complete!")
        print(f"{'='*60}")
        print(f"Location: {dataset.location}")
        print(f"Format: {format}")
        
        # Count images
        dataset_path = Path(dataset.location)
        train_images = len(list((dataset_path / "train").glob("*.jpg"))) if (dataset_path / "train").exists() else 0
        valid_images = len(list((dataset_path / "valid").glob("*.jpg"))) if (dataset_path / "valid").exists() else 0
        test_images = len(list((dataset_path / "test").glob("*.jpg"))) if (dataset_path / "test").exists() else 0
        
        print(f"\nDataset Statistics:")
        print(f"  Train: {train_images} images")
        print(f"  Valid: {valid_images} images")
        print(f"  Test: {test_images} images")
        print(f"  Total: {train_images + valid_images + test_images} images")
        print(f"{'='*60}\n")
        
        return dataset
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
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
        format=args.format,
        output_dir=args.output
    )
    
    print("Ready to train! Run:")
    print(f"  python -m src train --config configs/train_plate_detector.yaml")


if __name__ == "__main__":
    main()
