# YOLOv8 Training Results

Comprehensive training log for license plate detection using YOLOv8n with custom augmentation pipeline.

---

## 📊 Final Results Summary

### Initial Training (50 epochs)
**Date:** 2026-02-15  
**Platform:** Lightning.ai (Tesla T4 GPU, 15GB VRAM)  
**Training Time:** 49.2 minutes  
**Config:** [`configs/train_yolo.yaml`](configs/train_yolo.yaml)

| Metric | Value | Industry Standard | Status |
|--------|-------|-------------------|--------|
| **mAP@0.5** | **99.43%** | 95%+ | ✅ Excellent |
| **mAP@0.5:0.95** | **73.67%** | 70%+ | ✅ Good |
| **Precision** | **99.06%** | 90%+ | ✅ Excellent |
| **Recall** | **99.16%** | 90%+ | ✅ Excellent |
| **Inference Speed** | 2.6ms | <10ms | ✅ Real-time |

**Model:** YOLOv8n (3.01M params, 8.2 GFLOPs)  
**Best Checkpoint:** [`runs/detect/yolo/train/weights/best.pt`](runs/detect/yolo/train/weights/best.pt)

---

## 📈 Training Progress

### Epoch-by-Epoch Metrics

| Epoch | box_loss | cls_loss | dfl_loss | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|----------|----------|----------|---------|--------------|-----------|--------|
| 1 | 1.228 | 2.355 | 1.150 | 98.7% | 67.3% | 98.3% | 95.9% |
| 10 | 1.045 | 0.501 | 1.012 | 99.5% | 73.0% | 99.5% | 99.0% |
| 20 | 1.001 | 0.437 | 0.996 | 99.4% | 73.5% | 99.4% | 99.5% |
| 30 | 0.953 | 0.397 | 0.985 | 99.4% | 74.1% | 99.3% | 99.6% |
| 40 | 0.909 | 0.369 | 0.972 | 99.4% | 74.5% | 99.6% | 99.6% |
| **50** | **0.837** | **0.308** | **0.957** | **99.43%** | **73.67%** | **99.06%** | **99.16%** |

**Key Observations:**
- Fast convergence in first 10 epochs (mAP@0.5: 95.9% → 99.5%)
- Mosaic disabled at epoch 40 → further loss reduction
- Stable plateau from epoch 30-50

---

## 🆚 Comparison: Custom Implementation vs Baseline

### vs Colab Simple Training

| Metric | **My Implementation** | Colab Notebook | Improvement |
|--------|----------------------|----------------|-------------|
| mAP@0.5 | 99.43% | 99.5% | -0.07% (tie) |
| mAP@0.5:0.95 | 73.67% | 75.1% | -1.43% |
| Training Time | 49.2 min | 97.9 min | **50% faster** |
| Code Quality | Production | Notebook | ✅ Better |
| Custom Components | CBAM, Mosaic, MixUp | None | ✅ More skills |

**Conclusion:** Achieved comparable accuracy with half the training time, plus production-ready architecture.

### vs FasterRCNN-MobileNetV3 (Baseline)

| Metric | YOLOv8n | FasterRCNN | Improvement |
|--------|---------|------------|-------------|
| mAP@0.5 | **99.43%** | 72% | **+27.43%** 🚀 |
| Inference Speed | 2.6ms | 45ms | **17x faster** |
| Model Size | 6.3MB | 28MB | **4.4x smaller** |

---

## 🚀 Fine-Tuning (Round 2)

### Config: Lighter Augmentation (20 epochs)
**File:** [`configs/train_yolo_finetune.yaml`](configs/train_yolo_finetune.yaml)  
**Resume From:** `runs/detect/yolo/train/weights/best.pt`  

**Command:**
```bash
python -m src train --model yolo \
  --config configs/train_yolo_finetune.yaml \
  --resume runs/detect/yolo/train/weights/best.pt \
  --device cuda
```

**Expected:** mAP@0.5:0.95: 73.67% → **75%+**

---

## 📦 Trained Model Downloads

| Model | mAP@0.5 | Size | Download |
|-------|---------|------|----------|
| **YOLOv8n (best)** | 99.43% | 6.3MB | [best.pt](runs/detect/yolo/train/weights/best.pt) |

---

**Last Updated:** 2026-02-15
