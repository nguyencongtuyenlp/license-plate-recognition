# ALPR Demo Results

Inference demonstrations using trained YOLOv8 model on real traffic videos.

## 📊 Demo Videos

### Video 1: Traffic Scene
- **Input:** `video.mp4` (417 frames, 27 FPS)
- **Output:** `output_alpr.mp4`
- **Results:**
  - Vehicles detected: 17
  - Processing speed: 3.3 FPS
  - Total processing time: 126.3s

![Preview](output_alpr_preview.jpg)

### Video 2: Traffic Scene 2
- **Input:** `video2.mp4`
- **Output:** `output_alpr_video2.mp4`
- **Results:** (To be updated after processing)

## 🎯 Pipeline Components

1. **Vehicle Detection:** YOLOv8n (COCO pretrained)
2. **Tracking:** SORT algorithm
3. **Plate Detection:** YOLOv8n (Custom trained, 99.43% mAP@0.5)
4. **OCR:** EasyOCR

## 🚀 How to Reproduce

```bash
python demo_full_pipeline.py \
  --video video.mp4 \
  --output output_alpr.mp4 \
  --device cuda
```

See [`DEMO_GUIDE.md`](../DEMO_GUIDE.md) for detailed instructions.
