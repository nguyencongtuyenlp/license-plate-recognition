# Lightning.ai Training Guide

Complete guide for training the ALPR model on Lightning.ai's free GPU tier.

## Why Lightning.ai?

- **Free GPU**: T4 (16GB VRAM) available on free tier
- **Fast Training**: ~1.5-2 hours for 50 epochs (vs 6-8 hours on GTX 1650)
- **No Setup**: Pre-configured PyTorch environment
- **Cloud Storage**: Save checkpoints to cloud

---

## Step 1: Create Lightning.ai Studio

1. Go to [Lightning.ai](https://lightning.ai)
2. Sign up / Log in
3. Create new Studio:
   - **Name**: ALPR Training
   - **GPU**: T4 (16GB) - Free tier
   - **Environment**: PyTorch 2.0+

---

## Step 2: Clone Repository

```bash
# Clone your GitHub repo
git clone https://github.com/<your-username>/license_plate.git
cd license_plate
```

---

## Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA Available: True
```

---

## Step 4: Download Dataset

```bash
# Download dataset from Roboflow (COCO format)
python scripts/download_dataset.py --format coco --output data

# This will download ~8,255 images (~400MB)
# Takes about 2-3 minutes
```

**Verify dataset:**
```bash
ls data/coco/
# Should see: train/ valid/ test/ _annotations.coco.json
```

---

## Step 5: Configure Training

Edit `configs/train_plate_detector.yaml` for T4 GPU:

```yaml
training:
  batch_size: 16          # T4 can handle larger batches
  epochs: 50
  lr: 0.005
  amp_enabled: true       # Enable Mixed Precision for speed

device: "cuda"
```

---

## Step 6: Start Training

```bash
# Start training
python -m src train --config configs/train_plate_detector.yaml --device cuda

# Training will start and show progress:
# Epoch [0/50] Batch [0/360] Loss: 1.2345 LR: 0.001667
```

**What to expect:**
- **Duration**: ~1.5-2 hours for 50 epochs
- **Checkpoints**: Saved to `checkpoints/best.pth`
- **Logs**: TensorBoard logs in `runs/train/`
- **mAP@0.5**: Should reach ~0.85-0.90 after 50 epochs

---

## Step 7: Monitor Training

### Option 1: Terminal Output

Watch the terminal for epoch progress:
```
Epoch [10/50] — Loss: 0.3456 | mAP@0.5: 0.7234 | LR: 0.004500 | Time: 125.3s
  ★ New best mAP@0.5: 0.7234 — Checkpoint saved!
```

### Option 2: TensorBoard

```bash
# In a new terminal
tensorboard --logdir runs/train --port 6006

# Access at: http://localhost:6006
```

**Metrics to watch:**
- `train/loss` - Should decrease steadily
- `val/mAP@0.5` - Should increase to ~0.85-0.90
- `train/lr` - Should follow warmup + cosine schedule

---

## Step 8: Download Checkpoints

After training completes:

```bash
# Best model is saved at:
checkpoints/best.pth

# Download to local machine via Lightning.ai UI
# Or use Lightning CLI to sync
```

---

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM error:

```yaml
# Reduce batch size in config
training:
  batch_size: 8  # or 4
```

### Slow Download

If dataset download is slow:

```bash
# Download manually and upload to Lightning.ai
# Then extract to data/coco/
```

### Training Stops

If training stops unexpectedly:

```bash
# Resume from checkpoint
python -m src train --config configs/train_plate_detector.yaml --resume-from checkpoints/epoch_10.pth
```

---

## Performance Comparison

| Environment | GPU | Batch Size | Time (50 epochs) | Cost |
|-------------|-----|------------|------------------|------|
| Local GTX 1650 | 4GB | 2-4 | 6-8 hours | Free |
| Lightning.ai T4 | 16GB | 16 | 1.5-2 hours | Free ✅ |
| Colab T4 | 16GB | 16 | 2-3 hours | Free (limited) |
| Local RTX 3060 | 12GB | 8-12 | 2-3 hours | Free |

**Winner**: Lightning.ai T4 - Best combination of speed and cost!

---

## Tips for Best Results

1. **Use Mixed Precision**: Keep `amp_enabled: true` for 2x speedup
2. **Monitor Early**: Check first 5 epochs - loss should drop quickly
3. **Early Stopping**: Training may stop early if mAP plateaus (patience=10)
4. **Save Checkpoints**: Download `best.pth` before closing Studio
5. **TensorBoard**: Use it to debug if training doesn't converge

---

## Next Steps After Training

1. **Evaluate**: Test on validation set
   ```bash
   python -m src evaluate --checkpoint checkpoints/best.pth --device cuda
   ```

2. **Inference**: Run on test video
   ```bash
   python -m src infer --video test.mp4 --output result.mp4 --device cuda
   ```

3. **Deploy**: Use FastAPI to serve model
   ```bash
   python -m src api --host 0.0.0.0 --port 8000
   ```

---

## Questions?

- Lightning.ai Docs: https://lightning.ai/docs
- PyTorch Docs: https://pytorch.org/docs
- Project Issues: https://github.com/<your-username>/license_plate/issues

Happy Training! 🚀
