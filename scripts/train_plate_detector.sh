#!/bin/bash
# Train License Plate Detector
# Usage: bash scripts/train_plate_detector.sh

echo "=== Training License Plate Detector ==="
echo "Config: configs/train_plate_detector.yaml"

python -m src train \
    --config configs/train_plate_detector.yaml \
    "$@"
