import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset.dataset import DeepfakeDataset
from src.models.videomae import VideoMAEClassifier
from src.losses.ssl_loss import ConsistencyLoss
from torch.utils.data import ConcatDataset


# ── COMBINED dataset: FF++ balanced + DINOv2 diverse deepfakes ─────────────────
# data/faces:     1001 real + 1000 fake  (FaceForensics++ classic fakes)
# data/faces_dino: 358 real + 3058 fake  (diverse modern deepfakes)
# Combined:       1359 real + 4058 fake
ds1 = DeepfakeDataset("data/faces",      multi_view=True)
ds2 = DeepfakeDataset("data/faces_dino", multi_view=True)
dataset = ConcatDataset([ds1, ds2])
print(f"Combined dataset: {len(ds1)} (FF++) + {len(ds2)} (DINOv2) = {len(dataset)} total")

# train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
ACCUM_STEPS  = 4   # effective batch = 2×4 = 8


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# model
model = VideoMAEClassifier().to(device)


# Class weights for imbalanced combined dataset (1359 real vs 4058 fake → ratio ~3:1)
# Real class gets 3.0x weight so model doesn't ignore minority class
class_weights = torch.tensor([3.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
ssl_criterion = ConsistencyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# AMP — Automatic Mixed Precision: ~2x faster, ~40% less VRAM
scaler = torch.cuda.amp.GradScaler()


# 30 epochs — combined dataset is larger, early stopping handles convergence
epochs = 30
BEST_CKPT = "outputs/checkpoints/best_model.pth"

# tracking best validation accuracy
best_val_acc = 0.0
patience_counter = 0
PATIENCE = 8   # Stop early if no improvement for 8 epochs

os.makedirs("outputs/checkpoints", exist_ok=True)

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

    for step, (x_weak, x_strong, y) in enumerate(train_loop):
        x_weak   = x_weak.to(device)
        x_strong = x_strong.to(device)
        y        = y.to(device)

        with torch.cuda.amp.autocast():
            logits_weak,   embed_weak   = model(x_weak)
            logits_strong, embed_strong = model(x_strong)
            loss_cls = (criterion(logits_weak, y) + criterion(logits_strong, y)) / 2.0
            loss_ssl = ssl_criterion(embed_weak, embed_strong)
            loss     = (loss_cls + 0.5 * loss_ssl) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item(), cls=loss_cls.item(), ssl=loss_ssl.item())

    avg_train_loss = total_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
    
    with torch.no_grad():
        for x_weak, x_strong, y in val_loop:
            # We use weak view for standard validation
            x_weak = x_weak.to(device)
            y = y.to(device)
            
            logits_weak, _ = model(x_weak)
            loss = criterion(logits_weak, y)
            
            val_loss += loss.item()
            preds = torch.argmax(logits_weak, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best checkpoint & Early Stopping
    if val_acc > best_val_acc:
        print(f"--> Val Acc improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving best model...")
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"    ✔ Saved to {BEST_CKPT}")
        patience_counter = 0  # Reset counter
    else:
        patience_counter += 1
        print(f"--> No improvement. Early stopping counter: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n🚨 Early stopping triggered at epoch {epoch+1}.")
            break

