# YOLOv8 Training Summary

**Date:** 2026-02-15
**Platform:** Lightning.ai (Tesla T4 GPU)

## Models

### YOLOv8n Baseline
- **Path:** `training_artifacts/yolov8n_baseline/best.pt`
- **Metrics:** See TRAINING_RESULTS.md in root directory
- **Training Config:** `training_artifacts/yolov8n_baseline/args.yaml`
- **Training Curves:** `training_artifacts/visualizations/training_curves.png`

## Files Structure
```
training_artifacts/
├── TRAINING_SUMMARY.md          # This file
├── yolov8n_baseline/
│   ├── best.pt                  # Best checkpoint (6.3MB)
│   ├── last.pt                  # Last checkpoint (6.3MB)
│   ├── results.csv              # Epoch-by-epoch metrics
│   └── args.yaml                # Training hyperparameters
└── visualizations/
    ├── training_curves.png      # Loss & mAP curves
    ├── confusion_matrix.png     # Confusion matrix
    ├── val_batch0_pred.jpg      # Sample predictions
    └── labels.jpg               # Dataset statistics
```

## Usage

### Load Model for Inference
```python
from ultralytics import YOLO
model = YOLO('training_artifacts/yolov8n_baseline/best.pt')
results = model.predict('image.jpg')
```

### Resume Training
```bash
python -m src train --model yolo \
  --resume training_artifacts/yolov8n_baseline/best.pt \
  --epochs 100
```
