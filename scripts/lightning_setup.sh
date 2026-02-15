#!/bin/bash
# ============================================================
# Lightning.ai Quick Start Script
# ============================================================
# This script automates the setup process on Lightning.ai
# Run this after creating your T4 GPU studio
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "🚀 ALPR Training Setup - Lightning.ai"
echo "============================================================"
echo ""

# Step 1: Verify CUDA
echo "📋 Step 1/6: Verifying CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✅ GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Step 2: Install dependencies
echo "📦 Step 2/6: Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Step 3: Download dataset
echo "📥 Step 3/6: Downloading dataset from Roboflow..."
python scripts/download_dataset.py --format coco --output data
echo "✅ Dataset downloaded"
echo ""

# Step 4: Verify dataset
echo "🔍 Step 4/6: Verifying dataset structure..."
if [ -d "data/coco/train" ] && [ -d "data/coco/valid" ] && [ -d "data/coco/test" ]; then
    TRAIN_COUNT=$(ls data/coco/train/*.jpg 2>/dev/null | wc -l)
    VALID_COUNT=$(ls data/coco/valid/*.jpg 2>/dev/null | wc -l)
    TEST_COUNT=$(ls data/coco/test/*.jpg 2>/dev/null | wc -l)
    echo "✅ Dataset verified:"
    echo "   - Train: $TRAIN_COUNT images"
    echo "   - Valid: $VALID_COUNT images"
    echo "   - Test: $TEST_COUNT images"
else
    echo "❌ Dataset structure incomplete!"
    exit 1
fi
echo ""

# Step 5: Create directories
echo "📁 Step 5/6: Creating output directories..."
mkdir -p checkpoints runs/train
echo "✅ Directories created"
echo ""

# Step 6: Display training command
echo "🎓 Step 6/6: Ready to train!"
echo ""
echo "============================================================"
echo "Run the following command to start training:"
echo ""
echo "  python -m src train --config configs/train_plate_detector.yaml --device cuda"
echo ""
echo "Or use this optimized command for T4 GPU:"
echo ""
echo "  python -m src train \\"
echo "    --config configs/train_plate_detector.yaml \\"
echo "    --batch-size 16 \\"
echo "    --device cuda"
echo ""
echo "============================================================"
echo "📊 Monitor training:"
echo "  - Terminal: Watch epoch progress"
echo "  - TensorBoard: tensorboard --logdir runs/train --port 6006"
echo "  - Logs: tail -f runs/train/metrics.csv"
echo ""
echo "⏱️  Expected time: ~1.5-2 hours (50 epochs with early stopping)"
echo "🎯 Target mAP@0.5: >0.85"
echo "============================================================"
echo ""
echo "✅ Setup complete! Happy training! 🚀"
