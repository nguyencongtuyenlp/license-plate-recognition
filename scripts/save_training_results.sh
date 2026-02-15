#!/bin/bash
# =============================================================================
# Save Training Results to GitHub
# Lưu model weights, training logs, và visualizations lên GitHub
# =============================================================================

set -e  # Exit on error

echo "📦 Saving YOLOv8 Training Results to GitHub..."

# 1. Tạo thư mục lưu trữ
echo "📁 Creating backup directories..."
mkdir -p training_artifacts/yolov8n_baseline
mkdir -p training_artifacts/yolov8n_finetune
mkdir -p training_artifacts/visualizations

# 2. Copy model weights (tìm tự động)
echo "💾 Copying model weights..."
# Tìm thư mục train mới nhất
TRAIN_DIR=$(find runs/detect -type d -name "train*" | sort -r | head -1)
if [ -z "$TRAIN_DIR" ]; then
    echo "❌ Cannot find training directory!"
    exit 1
fi

echo "Found training dir: $TRAIN_DIR"

cp $TRAIN_DIR/weights/best.pt training_artifacts/yolov8n_baseline/best.pt
cp $TRAIN_DIR/weights/last.pt training_artifacts/yolov8n_baseline/last.pt

# Copy fine-tune model nếu có
FINETUNE_DIR=$(find runs/detect -type d -name "train2" 2>/dev/null | head -1)
if [ -n "$FINETUNE_DIR" ]; then
    cp $FINETUNE_DIR/weights/best.pt training_artifacts/yolov8n_finetune/best.pt 2>/dev/null || true
fi

# 3. Copy training logs
echo "📊 Copying training logs..."
cp $TRAIN_DIR/results.csv training_artifacts/yolov8n_baseline/
cp $TRAIN_DIR/args.yaml training_artifacts/yolov8n_baseline/

# 4. Copy visualizations
echo "🎨 Copying visualizations..."
cp $TRAIN_DIR/results.png training_artifacts/visualizations/training_curves.png 2>/dev/null || true
cp $TRAIN_DIR/confusion_matrix.png training_artifacts/visualizations/ 2>/dev/null || true
cp $TRAIN_DIR/val_batch0_pred.jpg training_artifacts/visualizations/ 2>/dev/null || true
cp $TRAIN_DIR/labels.jpg training_artifacts/visualizations/ 2>/dev/null || true

# 5. Tạo summary file
echo "📝 Creating summary file..."
cat > training_artifacts/TRAINING_SUMMARY.md <<EOF
# YOLOv8 Training Summary

**Date:** $(date +%Y-%m-%d)
**Platform:** Lightning.ai (Tesla T4 GPU)

## Models

### YOLOv8n Baseline
- **Path:** \`training_artifacts/yolov8n_baseline/best.pt\`
- **Metrics:** See TRAINING_RESULTS.md in root directory
- **Training Config:** \`training_artifacts/yolov8n_baseline/args.yaml\`
- **Training Curves:** \`training_artifacts/visualizations/training_curves.png\`

## Files Structure
\`\`\`
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
\`\`\`

## Usage

### Load Model for Inference
\`\`\`python
from ultralytics import YOLO
model = YOLO('training_artifacts/yolov8n_baseline/best.pt')
results = model.predict('image.jpg')
\`\`\`

### Resume Training
\`\`\`bash
python -m src train --model yolo \\
  --resume training_artifacts/yolov8n_baseline/best.pt \\
  --epochs 100
\`\`\`
EOF

# 6. Git LFS setup (optional, cho model weights)
echo "🔧 Setting up Git LFS for model files..."
if command -v git-lfs &> /dev/null; then
    git lfs track "*.pt"
    git lfs track "*.pth"
    git add .gitattributes
else
    echo "⚠️  Git LFS not installed. Model files will be tracked normally."
    echo "   Install with: git lfs install"
fi

# 7. Add to Git
echo "🔄 Adding files to Git..."
git add training_artifacts/
git add TRAINING_RESULTS.md
git add configs/

# 8. Commit
echo "💾 Committing changes..."
git commit -m "Add YOLOv8 training artifacts and results

- Model weights: yolov8n baseline (99.43% mAP@0.5)
- Training logs and hyperparameters
- Visualizations: loss curves, confusion matrix, predictions
- Complete training documentation

Trained on Lightning.ai (Tesla T4, 49.2 minutes)
" || echo "No changes to commit"

# 9. Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Done! Training results saved to GitHub."
echo ""
echo "📁 View artifacts at: training_artifacts/"
echo "📊 View full results: TRAINING_RESULTS.md"
echo ""
