"""
calibrate.py - Temperature scaling on validation logits.

Learns a single scalar T via LBFGS minimising NLL on val set.
Saves T to outputs/temperature.json and reports ECE before/after.
"""
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BEST_CHECKPOINT, BATCH_SIZE, CLASSES, EPOCHS, IMG_SIZE,
    MODEL_NAME, NUM_CLASSES, NUM_WORKERS, OUTPUT_DIR,
    SEED, TEMPERATURE_PATH, USE_AMP, VAL_CSV, VAL_DIR,
    seed_everything,
)
from dataset import SkinLesionDataset, val_transforms


# =====================================================================
# ECE
# =====================================================================

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(ece)


# =====================================================================
# Collect logits
# =====================================================================

@torch.no_grad()
def collect_logits(model, device):
    """Collect logits, using cached embeddings when available for instant execution."""
    val_feats_path = os.path.join(OUTPUT_DIR, "val_embeddings.npy")
    val_labels_path = os.path.join(OUTPUT_DIR, "val_labels.npy")

    if os.path.isfile(val_feats_path) and os.path.isfile(val_labels_path):
        print("Using cached validation embeddings to compute exact logits...")
        X_val = torch.from_numpy(np.load(val_feats_path)).float().to(device)
        y_val = torch.from_numpy(np.load(val_labels_path)).long()
        logits = model.classifier(X_val).cpu()
        return logits, y_val

    # Fallback to DataLoader evaluation
    val_ds = SkinLesionDataset(VAL_CSV, VAL_DIR, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels, _ in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# =====================================================================
# Temperature scaling
# =====================================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit temperature T via LBFGS minimising NLL."""
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=300)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.item())


# =====================================================================
# Main
# =====================================================================

def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    print("Collecting validation logits ...")
    logits, labels = collect_logits(model, device)

    # Before calibration
    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, labels.numpy())
    print(f"ECE before calibration: {ece_before:.4f}")

    # Fit T
    T = fit_temperature(logits, labels)
    print(f"Optimal temperature:    {T:.4f}")

    # After calibration
    probs_after = torch.softmax(logits / T, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, labels.numpy())
    print(f"ECE after calibration:  {ece_after:.4f}")

    # Save
    result = {
        "temperature": round(T, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4)
    }
    with open(TEMPERATURE_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[OK] Saved to {TEMPERATURE_PATH}")


if __name__ == "__main__":
    main()
