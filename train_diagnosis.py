"""
Agentic DermaGuard – Diagnosis Agent Training
==============================================
EfficientNet-B0 via timm, inverse-frequency weighted CrossEntropy,
AdamW + CosineAnnealingLR, MPS-accelerated training.

Best checkpoint saved by Validation Macro F1-score.
"""

import multiprocessing
import os
import sys
import time
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ──── project imports ────
from dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_dataloaders,
)

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
DATA_ROOT   = "/Users/vyomvarshney2005/Desktop/health-training/Health/training"
EPOCHS      = 10
LR          = 3e-4
WEIGHT_DECAY = 1e-2
CKPT_PATH   = "best_diagnosis_agent.pth"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def compute_class_weights(label_counts: List[int], device: torch.device) -> torch.Tensor:
    """Inverse frequency weighting: w_c = N_total / (num_classes * N_c)."""
    total = sum(label_counts)
    weights = [total / (NUM_CLASSES * c) for c in label_counts]
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[train] Class weights: {dict(zip(CLASS_NAMES, [f'{x:.3f}' for x in weights]))}")
    return w


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[train] Using Apple MPS backend ⚡")
        return torch.device("mps")
    print("[train] MPS not available – falling back to CPU")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={correct/total:.4f}")

    return running_loss / total, correct / total


# ──────────────────────────────────────────────
# Validation loop
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    device = select_device()

    # ── Data ──
    train_loader, val_loader, train_counts = get_dataloaders(DATA_ROOT)

    # ── Model ──
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    model = model.to(device)
    print(f"[train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss with class weights ──
    class_weights = compute_class_weights(train_counts, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Training ──
    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_macro_f1={val_f1:.4f}  |  "
            f"lr={lr_now:.6f}  time={elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }
            torch.save(ckpt, CKPT_PATH)
            print(f"    ✅ Saved best checkpoint → {CKPT_PATH}  (F1={val_f1:.4f})")

    print(f"\n[train] Done. Best Validation Macro F1 = {best_f1:.4f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

