# Quick Demo Guide

## 🚀 Running the Demo

### Prerequisites
```bash
pip install ultralytics opencv-python
```

### 1. Single Image Detection
```bash
python demo.py --image test_image.jpg --output result.jpg
```

### 2. Video Processing
```bash
python demo.py --video input.mp4 --output output.mp4 --device cuda
```

### 3. Webcam (Real-time)
```bash
python demo.py --webcam
```

## 📊 Full ALPR Pipeline

For complete vehicle detection + tracking + OCR:

```bash
# Process video with full pipeline
python -m src infer \
  --video input.mp4 \
  --output output.mp4 \
  --config configs/inference.yaml \
  --device cuda
```

## 🎯 Quick Test with Ultralytics CLI

```bash
# Fastest way to test
yolo detect predict \
  model=training_artifacts/yolov8n_baseline/best.pt \
  source=test_image.jpg \
  save=True \
  conf=0.5
```

Results saved to `runs/detect/predict/`

---

**Model Path:**
- Local: `training_artifacts/yolov8n_baseline/best.pt`
- Lightning.ai: `runs/detect/runs/yolo/train/weights/best.pt`
