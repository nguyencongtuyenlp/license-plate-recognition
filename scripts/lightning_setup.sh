#!/bin/bash
# ============================================================
# Lightning.ai Quick Start Script
# ============================================================
# Run this after cloning the repo on Lightning.ai T4 GPU studio
# ============================================================

set -e

echo "============================================================"
echo "🚀 ALPR Training Setup - Lightning.ai"
echo "============================================================"
echo ""

# Step 1: Verify CUDA
echo "📋 Step 1/5: Verifying CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✅ GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Step 2: Install dependencies
echo "📦 Step 2/5: Installing dependencies..."
pip install -q -r requirements.txt
pip install -q roboflow
echo "✅ Dependencies installed"
echo ""

# Step 3: Download dataset
echo "📥 Step 3/5: Downloading dataset from Roboflow..."
rm -rf data/  # Clean start
python3 scripts/download_dataset.py --format coco --output data
echo ""

# Step 4: Verify dataset
echo "🔍 Step 4/5: Verifying dataset..."
python3 -c "
import os, sys
ok = True
for split in ['train', 'valid', 'test']:
    path = f'data/coco/{split}'
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.endswith(('.jpg','.jpeg','.png'))])
        ann = os.path.exists(os.path.join(path, '_annotations.coco.json'))
        print(f'  ✅ {split}: {count} images, annotations: {\"✅\" if ann else \"❌\"}')
        if count == 0: ok = False
    else:
        print(f'  ❌ {split}: not found')
        ok = False
if not ok:
    print('⚠ Dataset verification failed!')
    sys.exit(1)
print('✅ Dataset verified!')
"
echo ""

# Step 5: Ready
echo "============================================================"
echo "✅ Setup complete! Start training with:"
echo ""
echo "  python -m src train \\"
echo "    --config configs/train_plate_detector.yaml \\"
echo "    --batch-size 16 \\"
echo "    --device cuda"
echo ""
echo "⏱️  Expected: ~1.5-2 hours | 🎯 Target mAP@0.5: >0.85"
echo "============================================================"
