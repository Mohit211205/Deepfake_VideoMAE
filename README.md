# 🎭 Deepfake Detection — VideoMAE + Domain SSL

> **99.5% validation accuracy** | Live demo on Hugging Face Spaces | Human-in-the-loop feedback system

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/AtrriJi/deepfake-videomae)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)](https://pytorch.org)

---

## 📌 Overview

A state-of-the-art deepfake detection system that combines:
- **VideoMAE** (Masked Autoencoder for Video) as the backbone
- **Domain SSL** — Self-Supervised Learning with multi-view consistency loss
- **Test-Time Augmentation (TTA)** — dual-view inference (original + horizontal flip)
- **Human-in-the-loop feedback** — users can correct predictions, model learns over time
- **3-tier verdict** — REAL / SUSPICIOUS / FAKE with adaptive thresholds

---

## 🏗️ Architecture

```
Input Video
    │
    ▼
MTCNN Face Extractor → 16 face crops (224×224)
    │
    ▼
Multi-View Augmentation
 ├── Weak View  (slight flip/jitter)
 └── Strong View (JPEG compress + noise + grayscale + temporal shuffle)
    │
    ▼
VideoMAE Backbone (MCG-NJU/videomae-base)
    │
    ├── Classification Loss (CrossEntropy + class weights)
    └── Consistency Loss (SSL — weak/strong view alignment)
    │
    ▼
TTA at Inference (avg of original + flipped probabilities)
    │
    ▼
3-Tier Verdict:
  < 38%  → ✅ REAL
  38–55% → ⚠️  SUSPICIOUS
  > 55%  → 🎭 FAKE
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **99.50%** |
| ROC-AUC | **0.9991** |
| Fake Precision | **100.0%** |
| Fake Recall | **99.0%** |
| F1 Score | **0.9951** |
| False Positive Rate | **0.0%** |

Evaluated on 401-video held-out validation set (80/20 split, seed=42).

---

## 🗂️ Project Structure

```
Deepfake_VideoMAE_DomainSSL/
├── backend/
│   └── api.py              # FastAPI backend — predict + feedback endpoints
├── frontend/
│   ├── index.html          # Dashboard UI
│   ├── script.js           # Frontend logic + feedback system
│   └── style.css           # Glassmorphism dark UI
├── src/
│   ├── dataset/
│   │   └── dataset.py      # DeepfakeDataset — multi-view augmentation
│   ├── models/
│   │   └── videomae.py     # VideoMAEClassifier wrapper
│   ├── losses/
│   │   └── ssl_loss.py     # ConsistencyLoss (SSL objective)
│   ├── train.py            # Training loop (AMP + gradient accumulation)
│   └── eval.py             # Full evaluation — confusion matrix, AUC, F1
├── Dockerfile              # Production container
├── requirements.txt        # Dependencies
├── reorganize_dino_dataset.py  # DINOv2 → VideoMAE format converter
└── download_youtube_dataset.py # YouTube data scraper
```

---

## 🚀 Quick Start

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/Deepfake_VideoMAE_DomainSSL
cd Deepfake_VideoMAE_DomainSSL

conda create -n deepfake_env python=3.10
conda activate deepfake_env
pip install -r requirements.txt
```

### Training

```bash
# Standard dataset (data/faces/)
PYTHONPATH=. python src/train.py

# With DINOv2 dataset (harder, more diverse)
# First reorganize: python reorganize_dino_dataset.py
# Then train.py uses data/faces_dino/ automatically
```

### Evaluation

```bash
PYTHONPATH=. python src/eval.py
```

### Run Locally

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860
```

---

## 🔑 Key Innovations

### 1. Multi-View Consistency (Domain SSL)
Two augmented views of the same video are fed to the model simultaneously. The SSL loss forces the model to produce consistent embeddings regardless of augmentation — improving domain generalization.

### 2. Strong Augmentation Pipeline
The "strong view" applies:
- JPEG compression (40–85% quality) — simulates social media re-encoding
- Gaussian noise — simulates low-light/camera noise
- Random grayscale — forces texture-based (not color-based) detection
- Temporal frame shuffling — prevents temporal shortcut learning
- Vertical/horizontal flips

### 3. Adaptive Class Weights
Dataset imbalance handled via weighted CrossEntropy:
- BYOL dataset (1000R : 1000F) → weights `[1.0, 2.5]`
- DINOv2 dataset (358R : 3058F) → weights `[8.5, 1.0]`

### 4. Human Feedback Loop
- `/feedback` POST endpoint stores user corrections to `feedback_log.json`
- `/feedback/stats` GET endpoint shows model accuracy as rated by users
- Collected feedback can be used for incremental fine-tuning

---

## 🌐 Deployment

Deployed on **Hugging Face Spaces** (FastAPI + Docker):
- Model weights hosted separately at `AtrriJi/deepfake-videomae-weights`
- Weights downloaded dynamically at startup (avoids 1GB LFS limit)

---

## 📦 Dependencies

```
torch==2.2.2
transformers==4.38.2
fastapi
facenet-pytorch
opencv-python
numpy<2
huggingface_hub
```

---

## 👥 Team

Group 50 — CS671 Project  
IIT Kanpur
