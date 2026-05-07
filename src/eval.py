import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)
from src.dataset.dataset import DeepfakeDataset
from src.models.videomae import VideoMAEClassifier
import torch.nn.functional as F

print("=" * 60)
print("  DEEPFAKE DETECTOR — FULL EVALUATION REPORT")
print("=" * 60)

# ── Dataset ──────────────────────────────────────────────────
dataset = DeepfakeDataset("data/faces", multi_view=False)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_set = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
val_loader = DataLoader(val_set, batch_size=2, num_workers=2)
print(f"\n📊 Validation Set Size: {val_size} videos")
print(f"   (80/20 split from {len(dataset)} total, seed=42)\n")

# ── Model ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

model = VideoMAEClassifier().to(device)

state_dict = torch.load("outputs/checkpoints/best_model.pth", map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('.attention.query.bias', '.attention.q_bias')
    new_k = new_k.replace('.attention.value.bias', '.attention.v_bias')
    if '.attention.key.bias' in new_k:
        continue
    new_state_dict[new_k] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()
print("✅ Model loaded: outputs/checkpoints/best_model.pth\n")

# ── Inference ─────────────────────────────────────────────────
all_preds = []
all_labels = []
all_probs = []

print("Running inference on validation set...")
with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        logits, _ = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]  # fake probability
        preds = (probs > 0.35).long()            # threshold = 0.35

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  Processed {(i+1)*2}/{val_size} videos...")

print(f"\n{'=' * 60}")
print("  RESULTS")
print(f"{'=' * 60}")

# ── Metrics ───────────────────────────────────────────────────
labels_np  = np.array(all_labels)
preds_np   = np.array(all_preds)
probs_np   = np.array(all_probs)

acc = accuracy_score(labels_np, preds_np)
auc = roc_auc_score(labels_np, probs_np)
cm  = confusion_matrix(labels_np, preds_np)

print(f"\n📈 Overall Accuracy : {acc*100:.2f}%")
print(f"📉 ROC-AUC Score    : {auc:.4f}  (1.0 = perfect)")

print(f"\n📋 Confusion Matrix:")
print(f"              Pred: REAL  Pred: FAKE")
print(f"  True: REAL     {cm[0][0]:5d}       {cm[0][1]:5d}")
print(f"  True: FAKE     {cm[1][0]:5d}       {cm[1][1]:5d}")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)
fpr       = fp / (fp + tn + 1e-8)

print(f"\n📊 Per-Class Breakdown:")
print(f"  REAL  — Correctly identified  : {tn} / {tn+fp}")
print(f"  FAKE  — Correctly identified  : {tp} / {tp+fn}")
print(f"\n🎯 Fake Detection Metrics:")
print(f"  Precision (of fakes caught)   : {precision*100:.1f}%")
print(f"  Recall    (fakes not missed)  : {recall*100:.1f}%")
print(f"  F1 Score                      : {f1:.4f}")
print(f"  False Positive Rate           : {fpr*100:.1f}%  (real misclassified as fake)")

print(f"\n{'=' * 60}")
print("  Full Classification Report (sklearn)")
print(f"{'=' * 60}")
print(classification_report(labels_np, preds_np, target_names=["REAL", "FAKE"]))
print("=" * 60)
