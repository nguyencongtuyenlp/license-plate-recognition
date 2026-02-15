# ⚡ Lightning.ai Training Guide

Complete step-by-step guide to train the license plate detector on Lightning.ai's free T4 GPU.

---

## 📋 Prerequisites

- Lightning.ai account (free tier)
- GitHub repository URL: `https://github.com/nguyencongtuyenlp/license-plate-recognition.git`
- Roboflow API key (optional, for dataset download)

---

## 🚀 Step-by-Step Setup

### 1️⃣ Create Lightning.ai Studio

1. Go to [Lightning.ai](https://lightning.ai/)
2. Sign in with GitHub
3. Click **"New Studio"**
4. Configure:
   - **Name:** `ALPR Training`
   - **GPU:** **T4 (16GB)** ✅ (Free tier)
   - **Environment:** PyTorch 2.0+
5. Click **"Create Studio"**

⏱️ Wait ~1-2 minutes for studio to start

---

### 2️⃣ Clone Repository

Open terminal in Lightning.ai and run:

```bash
# Clone your repository
git clone https://github.com/nguyencongtuyenlp/license-plate-recognition.git
cd license-plate-recognition

# Verify you're in the right directory
pwd
ls -la
```

**Expected output:**
```
/teamspace/studios/this_studio/license-plate-recognition
README.md  configs/  data/  docs/  requirements.txt  scripts/  src/
```

---

### 3️⃣ Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected output:**
```
CUDA available: True
GPU: Tesla T4
```

---

### 4️⃣ Download Dataset

**Option 1: Automatic (Recommended)**

```bash
# Download COCO format dataset
python scripts/download_dataset.py --format coco --output data

# Verify download
ls -lh data/coco/
```

**Option 2: Manual**

If the script fails, download manually:

1. Visit: https://universe.roboflow.com/nguyn-cng-tuyn/my-first-project-usuhh-7gecz
2. Download version 2 in **COCO format**
3. Upload to Lightning.ai studio
4. Extract to `data/coco/`

**Verify dataset structure:**
```bash
tree data/coco/ -L 2
```

Expected:
```
data/coco/
├── train/
│   ├── *.jpg (5,756 images)
│   └── _annotations.coco.json
├── valid/
│   ├── *.jpg (1,640 images)
│   └── _annotations.coco.json
└── test/
    ├── *.jpg (859 images)
    └── _annotations.coco.json
```

---

### 5️⃣ Configure Training

Check the training config:

```bash
cat configs/train_plate_detector.yaml
```

**Recommended settings for T4 GPU:**

```yaml
# configs/train_plate_detector.yaml
data:
  root_dir: "data"
  batch_size: 16        # T4 can handle this
  num_workers: 4        # Parallel data loading

model:
  num_classes: 2        # background + license_plate

training:
  epochs: 50
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3
  patience: 10          # Early stopping

device: "cuda"          # Use GPU
amp_enabled: true       # Mixed precision for speed
```

---

### 6️⃣ Start Training

```bash
# Start training with monitoring
python -m src train --config configs/train_plate_detector.yaml --device cuda

# Alternative: Run in background with nohup
nohup python -m src train --config configs/train_plate_detector.yaml --device cuda > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

**Expected output:**
```
============================================================
Starting Training — 50 epochs
Device: cuda
LR: 0.005, Momentum: 0.9
Warmup: 3 epochs
Early Stopping Patience: 10
============================================================

Epoch [0/50] Batch [0/360] Loss: 1.2345 LR: 0.000167
Epoch [0/50] Batch [20/360] Loss: 0.9876 LR: 0.000167
...
Epoch [0/50] — Loss: 0.8234 | mAP@0.5: 0.1234 | LR: 0.000167 | Time: 234.5s
  ★ New best mAP@0.5: 0.1234 — Saved!
```

⏱️ **Training time:** ~1.5-2 hours for 50 epochs (with early stopping, usually stops around epoch 30-35)

---

### 7️⃣ Monitor Training

**Option 1: TensorBoard (Recommended)**

Open a new terminal and run:

```bash
cd license-plate-recognition
tensorboard --logdir runs/train --host 0.0.0.0 --port 6006
```

Then click the **TensorBoard** button in Lightning.ai interface.

**Metrics to watch:**
- `train/loss` — Should decrease steadily
- `val/mAP@0.5` — Should increase (target: >0.85)
- `train/lr` — Learning rate schedule

**Option 2: CSV Logs**

```bash
# View metrics in real-time
tail -f runs/train/metrics.csv

# Or use pandas
python -c "import pandas as pd; df = pd.read_csv('runs/train/metrics.csv'); print(df.tail(20))"
```

---

### 8️⃣ Check Results

After training completes:

```bash
# View final metrics
cat runs/train/metrics.csv | tail -20

# Check best checkpoint
ls -lh checkpoints/
```

**Expected files:**
```
checkpoints/
├── best.pth          # Best model (highest mAP)
└── last.pth          # Last epoch model
```

---

### 9️⃣ Evaluate Model

```bash
# Evaluate on test set
python -m src evaluate \
  --checkpoint checkpoints/best.pth \
  --config configs/train_plate_detector.yaml \
  --device cuda
```

**Expected output:**
```
Evaluation Results:
  mAP@0.5: 0.8734
  Precision: 0.9012
  Recall: 0.8456
  Total images: 859
```

---

### 🔟 Download Results

**Download trained model:**

```bash
# Create results archive
mkdir -p results
cp checkpoints/best.pth results/
cp -r runs/train results/tensorboard_logs
tar -czf alpr_training_results.tar.gz results/

# Download via Lightning.ai file browser
# Right-click on alpr_training_results.tar.gz → Download
```

---

## 📊 Recording Results

Create a results file to document your training:

```bash
# Create results document
cat > TRAINING_RESULTS.md << 'EOF'
# Training Results

## Environment
- **Platform:** Lightning.ai
- **GPU:** Tesla T4 (16GB VRAM)
- **Training Date:** [DATE]
- **Training Time:** [TIME] hours

## Dataset
- **Total Images:** 8,255
- **Train:** 5,756 images
- **Validation:** 1,640 images
- **Test:** 859 images
- **Classes:** 1 (license_plate)

## Hyperparameters
- **Model:** FasterRCNN-MobileNetV3-FPN
- **Optimizer:** SGD (momentum=0.9, weight_decay=5e-4)
- **Learning Rate:** 0.005 (warmup 3 epochs + cosine annealing)
- **Batch Size:** 16
- **Epochs:** 50 (early stopped at epoch [X])
- **Mixed Precision:** Enabled (FP16)

## Results
- **Best mAP@0.5:** [VALUE]
- **Final Precision:** [VALUE]
- **Final Recall:** [VALUE]
- **Best Epoch:** [NUMBER]

## Training Curve
[Paste TensorBoard screenshot or describe curve]

## Sample Predictions
[Add screenshots of predictions if available]

## Notes
[Any observations during training]
EOF

# Edit with your actual results
nano TRAINING_RESULTS.md
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size

```bash
python -m src train \
  --config configs/train_plate_detector.yaml \
  --batch-size 8 \
  --device cuda
```

### Issue: Dataset Download Fails

**Solution:** Download manually from Roboflow and upload to Lightning.ai

### Issue: Training Too Slow

**Check:**
```bash
# Verify GPU is being used
nvidia-smi

# Check data loading
# If num_workers=0, increase to 4
```

### Issue: Early Stopping Too Early

**Solution:** Increase patience

```bash
python -m src train \
  --config configs/train_plate_detector.yaml \
  --patience 15 \
  --device cuda
```

---

## 💡 Tips for Best Results

1. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Use tmux for long training:**
   ```bash
   tmux new -s training
   python -m src train --config configs/train_plate_detector.yaml --device cuda
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t training
   ```

3. **Save intermediate checkpoints:**
   - Best model is auto-saved when mAP improves
   - Check `checkpoints/` directory regularly

4. **Backup results:**
   - Download checkpoints periodically
   - Export TensorBoard logs

---

## 📈 Expected Performance

Based on similar datasets:

| Metric | Expected Range | Target |
|--------|---------------|--------|
| mAP@0.5 | 0.80 - 0.90 | >0.85 |
| Precision | 0.85 - 0.95 | >0.90 |
| Recall | 0.75 - 0.85 | >0.80 |
| Training Time | 1.5 - 2.5 hours | ~2 hours |

---

## ✅ Checklist

- [ ] Lightning.ai studio created with T4 GPU
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Dataset downloaded and verified
- [ ] Training config reviewed
- [ ] Training started
- [ ] TensorBoard monitoring setup
- [ ] Training completed successfully
- [ ] Model evaluated on test set
- [ ] Results documented
- [ ] Checkpoints downloaded
- [ ] Results pushed to GitHub

---

## 🚀 Next Steps After Training

1. **Update README.md** with actual results
2. **Add training curves** (TensorBoard screenshots)
3. **Test inference** on sample videos
4. **Push results to GitHub**
5. **Update portfolio** with project link

---

## 📞 Support

If you encounter issues:
1. Check Lightning.ai [documentation](https://lightning.ai/docs)
2. Review error logs: `cat training.log`
3. Check GPU status: `nvidia-smi`
4. Verify dataset: `ls -R data/coco/`

---

**Good luck with training! 🎯**
