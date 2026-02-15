"""
Dataset Download Script — Roboflow REST API
=============================================
Download license plate dataset using Roboflow REST API directly.
No dependency on roboflow SDK — just requests + zipfile.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --format coco --output data
"""

import argparse
import os
import sys
import zipfile
import shutil
import glob

try:
    import requests
except ImportError:
    print("Error: requests package not installed.")
    print("Install it with: pip install requests")
    sys.exit(1)


# ---- Config ----
ROBOFLOW_API_KEY = "EmY26Ik4IZIW8iOyGxVL"
WORKSPACE = "nguyn-cng-tuyn"
PROJECT = "my-first-project-usuhh-7gecz"
VERSION = 2


def get_download_url(api_key: str, workspace: str, project: str,
                     version: int, fmt: str) -> str:
    """Get direct download URL from Roboflow REST API."""
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}"
    params = {"api_key": api_key}

    print(f"  Requesting download link from Roboflow API...")
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()
    if "export" in data and "link" in data["export"]:
        return data["export"]["link"]
    else:
        # Fallback: try to generate export first
        print("  Export not ready, requesting generation...")
        resp = requests.post(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "export" in data and "link" in data["export"]:
            return data["export"]["link"]

    raise RuntimeError(f"Could not get download URL. API response: {data}")


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress indicator."""
    print(f"  Downloading dataset zip...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8192

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                bar_len = 40
                filled = int(bar_len * downloaded // total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end="", flush=True)

    print(f"\n  ✅ Downloaded {downloaded/1024/1024:.1f} MB")


def extract_and_organize(zip_path: str, target_dir: str) -> None:
    """Extract zip and organize into target_dir/train, valid, test."""
    print(f"  Extracting zip file...")

    # Extract to a temp directory first
    temp_dir = zip_path + "_extracted"
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    # Find train/valid/test inside extracted content
    os.makedirs(target_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        # Search for the split directory recursively
        found = None
        for root, dirs, files in os.walk(temp_dir):
            if os.path.basename(root) == split:
                found = root
                break

        if found:
            dest = os.path.join(target_dir, split)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(found, dest)
            print(f"  ✅ {split}/ → {dest}")
        else:
            print(f"  ⚠ {split}/ not found in zip")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.remove(zip_path)
    print(f"  ✅ Cleaned up temp files")


def count_images(directory: str) -> int:
    """Count image files in a directory."""
    if not os.path.exists(directory):
        return 0
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        count += len(glob.glob(os.path.join(directory, ext)))
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download license plate dataset from Roboflow"
    )
    parser.add_argument("--api-key", type=str, default=ROBOFLOW_API_KEY)
    parser.add_argument("--format", type=str, choices=["coco", "yolov8"],
                        default="coco")
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    target_dir = os.path.join(args.output, "coco")

    print(f"\n{'='*60}")
    print(f"📥 Downloading Dataset from Roboflow")
    print(f"{'='*60}")
    print(f"  Project:  {PROJECT}")
    print(f"  Version:  {VERSION}")
    print(f"  Format:   {args.format}")
    print(f"  Output:   {os.path.abspath(target_dir)}")
    print(f"{'='*60}\n")

    # Step 1: Get download URL
    try:
        download_url = get_download_url(
            args.api_key, WORKSPACE, PROJECT, VERSION, args.format
        )
        print(f"  ✅ Got download link\n")
    except Exception as e:
        print(f"\n  ❌ Failed to get download URL: {e}")
        print(f"\n  Manual fallback:")
        print(f"  1. Visit: https://universe.roboflow.com/{WORKSPACE}/{PROJECT}")
        print(f"  2. Download COCO format, version {VERSION}")
        print(f"  3. Extract to {target_dir}/")
        sys.exit(1)

    # Step 2: Download zip
    zip_path = os.path.join(args.output, "dataset.zip")
    os.makedirs(args.output, exist_ok=True)

    try:
        download_file(download_url, zip_path)
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print(f"\n  Try downloading manually with wget:")
        print(f'  wget -O dataset.zip "{download_url}"')
        sys.exit(1)

    # Step 3: Extract and organize
    print()
    extract_and_organize(zip_path, target_dir)

    # Step 4: Verify
    print(f"\n{'='*60}")
    print(f"📊 Dataset Statistics")
    print(f"{'='*60}")

    total = 0
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(target_dir, split)
        img_count = count_images(split_dir)
        total += img_count

        ann_file = os.path.join(split_dir, "_annotations.coco.json")
        has_ann = "✅" if os.path.exists(ann_file) else "❌"

        print(f"  {split:>5}: {img_count:>5} images  |  annotations: {has_ann}")

    print(f"  {'total':>5}: {total:>5} images")
    print(f"{'='*60}")

    if total > 0:
        print(f"\n✅ Dataset ready! Location: {os.path.abspath(target_dir)}")
        print(f"\n🚀 Start training:")
        print(f"  python -m src train --config configs/train_plate_detector.yaml --device cuda\n")
    else:
        print(f"\n⚠ WARNING: 0 images found. Check {target_dir}/")


if __name__ == "__main__":
    main()
