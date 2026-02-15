# 🚗 Vehicle Counting + License Plate Recognition (ALPR)

End-to-end **PyTorch** pipeline for vehicle counting and license plate recognition, built from scratch with custom training loop, OOP design patterns, and graph algorithms.

> **Purpose**: Demonstrate real AI engineering skills — custom PyTorch Dataset, training loop (AMP, gradient clipping, cosine scheduler), SORT tracker with Kalman filter, DFS/BFS graph algorithms, and a FastAPI service.

---

## 📐 System Architecture

```mermaid
graph TD
    A[Input Video] --> B[Vehicle Detector<br/>FasterRCNN - Pretrained COCO]
    B --> C[SORT Tracker<br/>Kalman Filter + Hungarian]
    C --> D[Line-Crossing Counter<br/>Cross Product Math]
    B --> E[Plate Detector<br/>FasterRCNN - Custom Trained]
    E --> F[OCR Engine<br/>EasyOCR + Regex]
    D --> G[Results JSON]
    F --> G
    G --> H[FastAPI Response]
    
    subgraph Training Pipeline
        I[COCO Dataset] --> J[Custom DataLoader]
        J --> K[Training Loop<br/>AMP + Grad Clip + Cosine LR]
        K --> L[mAP@0.5 Evaluation]
        L --> M[Checkpoint]
    end
    
    subgraph Graph Algorithms
        N[Track Graph — DFS<br/>Trajectory Reconstruction]
        O[Track Associator — BFS<br/>Re-identification]
        P[Hierarchical NMS — BFS<br/>Cluster-based Suppression]
        Q[ROI Tree — DFS<br/>Spatial Search]
    end
```

---

## 🏗️ Project Structure

```
license_plate/
├── configs/
│   └── train_plate_detector.yaml    # Training hyperparameters
├── data/coco/                       # Roboflow COCO dataset (8,255 images)
│   ├── train/                       # 5,756 images
│   ├── valid/                       # 1,640 images
│   └── test/                        # 859 images
├── src/
│   ├── data/                        # Custom Dataset + Transforms
│   ├── models/                      # ABC + Factory Pattern detectors
│   ├── training/                    # Training loop, mAP, metrics
│   ├── tracking/                    # SORT, DFS track graph, BFS associator
│   ├── counting/                    # Line-crossing counter
│   ├── ocr/                         # EasyOCR + regex postprocess
│   ├── postprocess/                 # BFS NMS, DFS ROI tree
│   ├── inference/                   # End-to-end pipeline
│   ├── api/                         # FastAPI service
│   └── utils/                       # Config (Singleton), Logger
├── scripts/                         # Shell scripts
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Skills Demonstrated

| Skill | Where | Details |
|-------|-------|---------|
| **OOP** | `models/base_detector.py` | ABC, Factory, Composition, Singleton |
| **DFS** | `tracking/track_graph.py`, `postprocess/roi_tree.py` | Trajectory reconstruction, spatial search |
| **BFS** | `tracking/associator.py`, `postprocess/nms.py` | Re-identification, hierarchical NMS |
| **Kalman Filter** | `tracking/sort_tracker.py` | State estimation (7D state vector) |
| **Linear Algebra** | `training/metrics.py`, `counting/line_counter.py` | IoU, cross product, matrix operations |
| **Deep Learning** | `training/trainer.py` | AMP, gradient clipping, cosine annealing |
| **Transfer Learning** | `models/plate_detector.py` | FasterRCNN backbone fine-tuning |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**Option 1: Automatic (Recommended)**

```bash
python scripts/download_dataset.py --format coco
```

**Option 2: Manual**

1. Visit [Roboflow Dataset](https://universe.roboflow.com/nguyn-cng-tuyn/my-first-project-usuhh-7gecz)
2. Download version 2 in COCO format
3. Extract to `data/coco/`

**Dataset Info:**
- **Source**: Roboflow
- **Images**: 8,255 (train: 5,756 | val: 1,640 | test: 859)
- **Classes**: 1 (license_plate)
- **Format**: COCO JSON

### 3. Train Plate Detector

```bash
python -m src train --config configs/train_plate_detector.yaml

# With overrides:
python -m src train --config configs/train_plate_detector.yaml --epochs 30 --batch-size 2 --device cuda
```

### 4. Evaluate

```bash
python -m src evaluate --checkpoint checkpoints/best.pth --device cpu
```

### 5. Run Inference on Video

```bash
python -m src infer --video path/to/video.mp4 --output output.mp4 --device cpu
```

### 6. Start API Server

```bash
python -m src api --host 0.0.0.0 --port 8000

# Test with curl:
curl -X POST http://localhost:8000/infer_video -F "video=@test_video.mp4"
```

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Model | FasterRCNN-MobileNetV3-FPN |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| LR Schedule | Warmup (3 epochs) + Cosine Annealing |
| Mixed Precision | Enabled (FP16 forward, FP32 backward) |
| Gradient Clipping | Max norm = 10.0 |
| Batch Size | 4 (GTX 1650 4GB) |
| Epochs | 50 (with early stopping, patience=10) |

---

## 💻 GPU Requirements

### Minimum (Works but slow)
- **GPU**: GTX 1650 (4GB VRAM)
- **Batch Size**: 2-4
- **Training Time**: ~6-8 hours (50 epochs)

### Recommended (Fast training)
- **GPU**: RTX 3060 (12GB VRAM) or better
- **Batch Size**: 8-16
- **Training Time**: ~2-3 hours (50 epochs)

### Lightning.ai Free Tier ⚡ (Best Option!)
- **GPU**: T4 (16GB VRAM) - **FREE**
- **Batch Size**: 16
- **Training Time**: ~1.5-2 hours (50 epochs)
- **Guide**: See [docs/lightning_ai_setup.md](docs/lightning_ai_setup.md)

---

## 🔮 Future Improvements

- [ ] Multi-class vehicle detection training (not just pretrained)
- [ ] DeepSORT with appearance features (Re-ID network)
- [ ] Vietnamese plate OCR fine-tuning
- [ ] Real-time streaming inference (WebSocket)
- [ ] Model quantization (INT8) for edge deployment
- [ ] Multi-camera tracking

---

## 📝 License

MIT License
