# 📊 Training Results

> **Status:** 🔄 Training in progress...
> 
> This document will be updated with actual training results from Lightning.ai

---

## 🖥️ Training Environment

| Parameter | Value |
|-----------|-------|
| **Platform** | Lightning.ai |
| **GPU** | Tesla T4 (16GB VRAM) |
| **CUDA Version** | [TO BE FILLED] |
| **PyTorch Version** | [TO BE FILLED] |
| **Training Date** | [TO BE FILLED] |
| **Training Duration** | [TO BE FILLED] hours |

---

## 📁 Dataset Information

| Metric | Value |
|--------|-------|
| **Source** | Roboflow |
| **Format** | COCO JSON |
| **Total Images** | 8,255 |
| **Training Set** | 5,756 images |
| **Validation Set** | 1,640 images |
| **Test Set** | 859 images |
| **Classes** | 1 (license_plate) |
| **Annotations** | Bounding boxes |

---

## ⚙️ Hyperparameters

```yaml
Model Architecture:
  - Backbone: FasterRCNN-MobileNetV3-FPN
  - Pretrained: COCO weights
  - Fine-tuned: License plate detection

Optimizer:
  - Type: SGD
  - Momentum: 0.9
  - Weight Decay: 5e-4

Learning Rate Schedule:
  - Initial LR: 0.005
  - Warmup: 3 epochs (linear)
  - Schedule: Cosine Annealing
  - Min LR: 1e-6

Training Configuration:
  - Batch Size: 16
  - Epochs: 50 (max)
  - Early Stopping: Patience 10
  - Mixed Precision: Enabled (FP16)
  - Gradient Clipping: Max norm 10.0

Data Augmentation:
  - Random Horizontal Flip (p=0.5)
  - Color Jitter (brightness, contrast, saturation)
  - Random Resize (800-1333px)
```

---

## 📈 Training Results

### Final Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Best mAP@0.5** | [TO BE FILLED] | Target: >0.85 |
| **Precision** | [TO BE FILLED] | Target: >0.90 |
| **Recall** | [TO BE FILLED] | Target: >0.80 |
| **F1-Score** | [TO BE FILLED] | Harmonic mean |
| **Best Epoch** | [TO BE FILLED] | When early stopping triggered |
| **Total Epochs** | [TO BE FILLED] | Actual epochs run |

### Training Curve

```
[TO BE FILLED: Paste TensorBoard screenshot or ASCII plot]

Example:
Epoch | Train Loss | Val mAP@0.5 | LR
------|------------|-------------|-------
  0   |   1.234    |    0.123    | 0.00017
  5   |   0.876    |    0.456    | 0.00500
 10   |   0.543    |    0.678    | 0.00450
 15   |   0.321    |    0.789    | 0.00350
 20   |   0.234    |    0.834    | 0.00200
 25   |   0.198    |    0.867    | 0.00100
 30   |   0.176    |    0.873    | 0.00050
```

### Loss Convergence

- **Initial Loss:** [TO BE FILLED]
- **Final Loss:** [TO BE FILLED]
- **Convergence:** [Smooth/Unstable/etc.]

---

## 🎯 Test Set Evaluation

```bash
# Command used:
python -m src evaluate \
  --checkpoint checkpoints/best.pth \
  --config configs/train_plate_detector.yaml \
  --device cuda
```

### Results

| Metric | Value |
|--------|-------|
| **Test mAP@0.5** | [TO BE FILLED] |
| **Test Precision** | [TO BE FILLED] |
| **Test Recall** | [TO BE FILLED] |
| **Inference Time** | [TO BE FILLED] ms/image |

---

## 📸 Sample Predictions

### Good Predictions

```
[TO BE FILLED: Add screenshots or descriptions of successful detections]

Example:
- Image: test_001.jpg
- Ground Truth: 1 plate
- Predicted: 1 plate
- IoU: 0.95
- Confidence: 0.98
```

### Challenging Cases

```
[TO BE FILLED: Add examples of difficult cases]

Example:
- Occlusion: Partial plate visibility
- Lighting: Low light conditions
- Angle: Extreme viewing angles
```

---

## 💾 Model Checkpoints

| File | Size | Description |
|------|------|-------------|
| `checkpoints/best.pth` | [TO BE FILLED] MB | Best validation mAP |
| `checkpoints/last.pth` | [TO BE FILLED] MB | Final epoch |

**Download:** [Link to Google Drive/Dropbox if needed]

---

## 📊 TensorBoard Logs

**Location:** `runs/train/`

**Key Metrics Tracked:**
- ✅ Training loss (per batch)
- ✅ Validation mAP@0.5 (per epoch)
- ✅ Learning rate schedule
- ✅ Gradient norms

**View logs:**
```bash
tensorboard --logdir runs/train --host 0.0.0.0 --port 6006
```

---

## 🔍 Observations & Notes

### What Worked Well

- [TO BE FILLED]
- Example: "Mixed precision training provided 1.5x speedup"
- Example: "Cosine annealing helped avoid local minima"

### Challenges Encountered

- [TO BE FILLED]
- Example: "Initial learning rate too high, caused instability"
- Example: "Dataset imbalance in lighting conditions"

### Lessons Learned

- [TO BE FILLED]
- Example: "Warmup period crucial for stable training"
- Example: "Batch size 16 optimal for T4 GPU"

---

## 🚀 Next Steps

- [ ] Fine-tune on additional data
- [ ] Experiment with different backbones (ResNet50, EfficientNet)
- [ ] Add data augmentation (Mosaic, MixUp)
- [ ] Optimize for inference speed
- [ ] Deploy to production API

---

## 📝 Training Log Excerpt

```
[TO BE FILLED: Paste relevant training log sections]

Example:
============================================================
Starting Training — 50 epochs
Device: cuda
LR: 0.005, Momentum: 0.9
Warmup: 3 epochs
Early Stopping Patience: 10
============================================================

Epoch [0/50] — Loss: 0.8234 | mAP@0.5: 0.1234 | Time: 234.5s
  ★ New best mAP@0.5: 0.1234 — Saved!

Epoch [10/50] — Loss: 0.4567 | mAP@0.5: 0.6789 | Time: 220.3s
  ★ New best mAP@0.5: 0.6789 — Saved!

...

⚠ Early Stopping! No improvement for 10 epochs.
Training complete! Total time: 5432.1s
Best mAP@0.5: 0.8734
============================================================
```

---

## 🏆 Comparison with Baselines

| Model | mAP@0.5 | Params | Speed (FPS) |
|-------|---------|--------|-------------|
| **Ours (MobileNetV3)** | [TO BE FILLED] | ~5M | [TO BE FILLED] |
| YOLOv5s | ~0.82 | 7M | ~60 |
| FasterRCNN-ResNet50 | ~0.88 | 41M | ~15 |

---

## 📧 Contact

**Author:** Nguyen Cong Tuyen  
**Email:** nguyencongtuyenlp@gmail.com  
**GitHub:** [@nguyencongtuyenlp](https://github.com/nguyencongtuyenlp)

---

**Last Updated:** [TO BE FILLED]  
**Training Status:** 🔄 In Progress / ✅ Complete
