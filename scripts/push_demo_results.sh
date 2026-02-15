#!/bin/bash
# =============================================================================
# Push Demo Results from Lightning.ai to GitHub
# Upload inference videos, screenshots, and results summary
# =============================================================================

set -e  # Exit on error

echo "📤 Pushing Demo Results to GitHub..."

# 1. Create demo results directory
echo "📁 Creating demo_results directory..."
mkdir -p demo_results

# 2. Copy demo videos
echo "🎬 Copying demo videos..."
cp output_alpr.mp4 demo_results/ 2>/dev/null || echo "⚠️  output_alpr.mp4 not found"
cp alpr_output.mp4 demo_results/ 2>/dev/null || echo "⚠️  alpr_output.mp4 not found"

# Find all generated videos
find runs/demo -name "*.mp4" -exec cp {} demo_results/ \; 2>/dev/null || true

# 3. Extract screenshots (first frame of each video)
echo "📸 Extracting screenshots..."
for video in demo_results/*.mp4; do
    if [ -f "$video" ]; then
        basename=$(basename "$video" .mp4)
        ffmpeg -i "$video" -vframes 1 -y "demo_results/${basename}_preview.jpg" 2>/dev/null || \
            echo "⚠️  Could not extract screenshot from $video (ffmpeg not available)"
    fi
done

# 4. Create demo summary
echo "📝 Creating demo summary..."
cat > demo_results/DEMO_RESULTS.md <<EOF
# ALPR Demo Results

Inference demonstrations using trained YOLOv8 model on real traffic videos.

## 📊 Demo Videos

### Video 1: Traffic Scene
- **Input:** \`video.mp4\` (417 frames, 27 FPS)
- **Output:** \`output_alpr.mp4\`
- **Results:**
  - Vehicles detected: 17
  - Processing speed: 3.3 FPS
  - Total processing time: 126.3s

![Preview](output_alpr_preview.jpg)

### Video 2: Traffic Scene 2
- **Input:** \`video2.mp4\`
- **Output:** \`output_alpr_video2.mp4\`
- **Results:** (To be updated after processing)

## 🎯 Pipeline Components

1. **Vehicle Detection:** YOLOv8n (COCO pretrained)
2. **Tracking:** SORT algorithm
3. **Plate Detection:** YOLOv8n (Custom trained, 99.43% mAP@0.5)
4. **OCR:** EasyOCR

## 🚀 How to Reproduce

\`\`\`bash
python demo_full_pipeline.py \\
  --video video.mp4 \\
  --output output_alpr.mp4 \\
  --device cuda
\`\`\`

See [\`DEMO_GUIDE.md\`](../DEMO_GUIDE.md) for detailed instructions.
EOF

# 5. Git operations
echo "🔄 Adding files to Git..."
git add demo_results/

echo "💾 Committing changes..."
git commit -m "Add ALPR demo results

- Demo videos with inference results
- Preview screenshots
- Processing statistics and summary
" || echo "No changes to commit"

echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Done! Demo results pushed to GitHub."
echo ""
echo "📁 View at: demo_results/"
echo "📄 Summary: demo_results/DEMO_RESULTS.md"
echo ""
