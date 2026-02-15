#!/bin/bash
# Run Demo Inference on Video
# Usage: bash scripts/run_demo.sh <video_path>

VIDEO=${1:-"sample_video.mp4"}
OUTPUT="output_demo.mp4"

echo "=== Running ALPR Demo ==="
echo "Input: $VIDEO"
echo "Output: $OUTPUT"

python -m src infer \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --checkpoint checkpoints/best.pth \
    --device cpu \
    "$@"
