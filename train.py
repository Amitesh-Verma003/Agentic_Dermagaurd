"""
train.py - EfficientNet-B0 on HAM10000 with class-weighted loss.

- If GPU is available: Fine-tune timm efficientnet_b0 with AdamW, cosine LR,
  mixed precision, and inverse-sqrt class weighting.
- If CPU only: Extract frozen timm embeddings (cached to .npy) with no_grad,
  train LogisticRegression(class_weight="balanced"), evaluate validation metrics,
  and export the classifier into the PyTorch model saved at outputs/best.pt.
"""
import argparse
import math
import os
import time

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import (
    BEST_CHECKPOINT, BATCH_SIZE, CLASSES, CLASS2IDX, EPOCHS, IMG_SIZE,
    LR, LABEL_SMOOTHING, MODEL_NAME, NUM_CLASSES, NUM_WORKERS,
    OUTPUT_DIR, SEED, TRAIN_CSV, TRAIN_DIR, USE_AMP, VAL_CSV, VAL_DIR,
    WARMUP_EPOCHS, WEIGHT_DECAY, seed_everything,
)
from dataset import SkinLesionDataset, train_transforms, val_transforms


# =====================================================================
# Feature Extraction (CPU Workflow)
# =====================================================================

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract pooled embeddings from frozen timm backbone."""
    model.eval()
    features_list = []
    labels_list = []
    image_ids_list = []

    total_batches = len(loader)
    print(f"  Extracting features ({total_batches} batches)...")

    for i, (imgs, labels, img_ids) in enumerate(loader):
        imgs = imgs.to(device)
        feats = model(imgs)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
        image_ids_list.extend(img_ids)
        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            print(f"    Batch [{i + 1}/{total_batches}] complete")

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    return features, labels, image_ids_list


def train_cpu_workflow():
    print("=" * 60)
    print("  CPU Mode: Frozen Backbone Embedding + Logistic Regression")
    print("=" * 60)

    device = torch.device("cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Datasets & Loaders
    # For feature extraction, use val_transforms (deterministic resize+norm)
    train_ds = SkinLesionDataset(TRAIN_CSV, TRAIN_DIR, val_transforms)
    val_ds = SkinLesionDataset(VAL_CSV, VAL_DIR, val_transforms)

    batch_size = 32  # larger batch size for fast inference
    num_workers = 0  # avoid Windows multiprocessing spawn overhead

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    train_feats_path = os.path.join(OUTPUT_DIR, "train_embeddings.npy")
    train_labels_path = os.path.join(OUTPUT_DIR, "train_labels.npy")
    val_feats_path = os.path.join(OUTPUT_DIR, "val_embeddings.npy")
    val_labels_path = os.path.join(OUTPUT_DIR, "val_labels.npy")

    # 2. Extract or Load Cached Embeddings
    if os.path.isfile(train_feats_path) and os.path.isfile(train_labels_path):
        print(f"> Loading cached train embeddings from {train_feats_path}...")
        X_train = np.load(train_feats_path)
        y_train = np.load(train_labels_path)
    else:
        print(f"> Building frozen {MODEL_NAME} feature extractor...")
        feat_extractor = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        feat_extractor.to(device).eval()

        print("> Extracting train set embeddings...")
        X_train, y_train, _ = extract_features(feat_extractor, train_loader, device)
        np.save(train_feats_path, X_train)
        np.save(train_labels_path, y_train)
        print(f"  [OK] Saved train embeddings: shape {X_train.shape}")

    if os.path.isfile(val_feats_path) and os.path.isfile(val_labels_path):
        print(f"> Loading cached val embeddings from {val_feats_path}...")
        X_val = np.load(val_feats_path)
        y_val = np.load(val_labels_path)
    else:
        print("> Extracting validation set embeddings...")
        feat_extractor = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        feat_extractor.to(device).eval()
        X_val, y_val, _ = extract_features(feat_extractor, val_loader, device)
        np.save(val_feats_path, X_val)
        np.save(val_labels_path, y_val)
        print(f"  [OK] Saved val embeddings: shape {X_val.shape}")

    # 3. Train Logistic Regression
    print("\n> Training LogisticRegression(class_weight='balanced', max_iter=1000)...")
    t0 = time.time()
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"  [OK] LogisticRegression fit completed in {fit_time:.2f}s")

    # 4. Evaluate Validation Set
    val_preds = clf.predict(X_val)
    val_acc = float((val_preds == y_val).mean())
    val_bal_acc = float(balanced_accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

    mel_idx = CLASS2IDX["mel"]
    val_mel_recall = float(recall_score(y_val, val_preds, labels=[mel_idx],
                                        average=None, zero_division=0)[0])

    print("\n" + "=" * 60)
    print("  Validation Results (CPU Baseline)")
    print("=" * 60)
    print(f"  Accuracy:          {val_acc:.4f}")
    print(f"  Balanced Accuracy: {val_bal_acc:.4f}")
    print(f"  Macro F1:          {val_macro_f1:.4f}")
    print(f"  Melanoma Recall:   {val_mel_recall:.4f}")
    print("=" * 60)

    # 5. Pack into standard PyTorch timm model
    print(f"\n> Packaging weights into PyTorch {MODEL_NAME} checkpoint...")
    full_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    
    # In timm efficientnet_b0, classifier head is full_model.classifier
    # weight: (num_classes, num_features), bias: (num_classes)
    weight_tensor = torch.from_numpy(clf.coef_).float()
    bias_tensor = torch.from_numpy(clf.intercept_).float()

    with torch.no_grad():
        full_model.classifier.weight.copy_(weight_tensor)
        full_model.classifier.bias.copy_(bias_tensor)

    torch.save({
        "model": full_model.state_dict(),
        "epoch": 1,
        "best_f1": val_macro_f1,
        "mode": "logistic_regression_head",
    }, BEST_CHECKPOINT)

    # Also save last.pt for resumability compatibility
    last_ckpt = os.path.join(OUTPUT_DIR, "last.pt")
    torch.save({
        "model": full_model.state_dict(),
        "epoch": 1,
        "best_f1": val_macro_f1,
    }, last_ckpt)

    print(f"[OK] Checkpoint saved successfully to {BEST_CHECKPOINT}")
    return val_macro_f1


# =====================================================================
# GPU Fine-Tuning Workflow
# =====================================================================

def compute_class_weights(dataset: SkinLesionDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    freqs = np.array([counts[c] for c in CLASSES], dtype=np.float64)
    weights = 1.0 / np.sqrt(freqs + 1e-6)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def cosine_lr_with_warmup(optimizer, epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def validate_gpu(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / n
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    mel_idx = CLASS2IDX["mel"]
    mel_recall = recall_score(all_labels, all_preds, labels=[mel_idx],
                              average=None, zero_division=0)[0]
    return {
        "loss": total_loss / n,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "melanoma_recall": mel_recall,
    }


def train_gpu_workflow(resume=False):
    device = torch.device("cuda")
    print(f"Device: {device} -- {torch.cuda.get_device_name()}")

    train_ds = SkinLesionDataset(TRAIN_CSV, TRAIN_DIR, train_transforms)
    val_ds   = SkinLesionDataset(VAL_CSV, VAL_DIR, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)
    weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    start_epoch = 0
    best_f1 = 0.0

    last_ckpt = os.path.join(OUTPUT_DIR, "last.pt")
    if resume and os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"> Resumed from epoch {start_epoch}, best_f1={best_f1:.4f}")

    header = (f"{'Ep':>3} {'LR':>9} {'TrLoss':>8} "
              f"{'VLoss':>8} {'Acc':>6} {'BAcc':>6} {'F1':>6} {'MelR':>6} {'Time':>6}")
    print("\n" + header)
    print("-" * len(header))

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        lr = cosine_lr_with_warmup(optimizer, epoch, EPOCHS, WARMUP_EPOCHS, LR)
        model.train()
        running_loss, n = 0.0, 0

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = running_loss / n
        val = validate_gpu(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        marker = ""
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "best_f1": best_f1},
                       BEST_CHECKPOINT)
            marker = " *"

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1},
                   last_ckpt)

        print(f"{epoch:>3} {lr:>9.2e} {train_loss:>8.4f} "
              f"{val['loss']:>8.4f} {val['accuracy']:>6.3f} "
              f"{val['balanced_accuracy']:>6.3f} {val['macro_f1']:>6.3f} "
              f"{val['melanoma_recall']:>6.3f} {elapsed:>5.0f}s{marker}")

    print(f"\n[OK] Training done. Best val macro-F1: {best_f1:.4f}")
    print(f"   Checkpoint: {BEST_CHECKPOINT}")


# =====================================================================
# Main Dispatcher
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU feature extraction workflow")
    args = parser.parse_args()

    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if torch.cuda.is_available() and not args.force_cpu:
        train_gpu_workflow(resume=args.resume)
    else:
        train_cpu_workflow()


if __name__ == "__main__":
    main()
