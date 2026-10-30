"""
evaluate_diagnosis.py - Full evaluation on the validation set.

Outputs:
  outputs/diagnosis_val.parquet
  outputs/diagnosis_metrics.json
  outputs/confusion_matrix.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score,
)
import torch
from torch.utils.data import DataLoader

from config import (
    BEST_CHECKPOINT, CLASSES, CLASS2IDX, IDX2CLASS, NUM_CLASSES,
    OUTPUT_DIR, SEED, TEMPERATURE_PATH, VAL_CSV, VAL_DIR,
    BATCH_SIZE, NUM_WORKERS, COL_IMAGE_ID, COL_LABEL, seed_everything,
)
from dataset import SkinLesionDataset, val_transforms
from calibrate import expected_calibration_error

# Add project root to path so agent import works
sys.path.insert(0, os.path.dirname(__file__))
from agents.diagnosis_agent import DiagnosisAgent


def main():
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize DiagnosisAgent
    print("> Initializing DiagnosisAgent with calibrated checkpoint...")
    agent = DiagnosisAgent(BEST_CHECKPOINT, TEMPERATURE_PATH)
    print(f"  [OK] DiagnosisAgent initialized (temperature: {agent.temperature:.4f})")

    # 2. Load val dataset metadata
    val_ds = SkinLesionDataset(VAL_CSV, VAL_DIR, val_transforms)
    print(f"> Validation set contains {len(val_ds):,} images.")

    # Smoke test agent.predict and agent.predict_batch
    sample_path = val_ds.df["filepath"].iloc[0]
    sample_pred = agent.predict(sample_path)
    print(f"  Sanity test agent.predict(): {sample_path} -> {sample_pred['label']} ({sample_pred['confidence']:.2%})")

    batch_test = agent.predict_batch([val_ds.df["filepath"].iloc[0], val_ds.df["filepath"].iloc[1]])
    print(f"  Sanity test agent.predict_batch(): processed {len(batch_test)} samples successfully.")

    # 3. Compute predictions across validation set
    val_feats_path = os.path.join(OUTPUT_DIR, "val_embeddings.npy")
    val_labels_path = os.path.join(OUTPUT_DIR, "val_labels.npy")

    if os.path.isfile(val_feats_path) and os.path.isfile(val_labels_path):
        print("\n> Evaluating using cached validation features through calibrated model head...")
        X_val = torch.from_numpy(np.load(val_feats_path)).float().to(agent.device)
        y_val_indices = np.load(val_labels_path)
        with torch.no_grad():
            logits = agent.model.classifier(X_val)
            probs = torch.softmax(logits / agent.temperature, dim=1)
        probs_np = probs.cpu().numpy()
        image_ids = val_ds.df[COL_IMAGE_ID].tolist()
        true_labels = [IDX2CLASS[idx] for idx in y_val_indices]
    else:
        print("\n> Evaluating validation images through full model...")
        loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        probs_list = []
        image_ids = []
        true_labels = []
        for imgs, labels, iids in loader:
            imgs = imgs.to(agent.device)
            with torch.no_grad():
                logits = agent.model(imgs)
                probs = torch.softmax(logits / agent.temperature, dim=1)
            probs_list.append(probs.cpu().numpy())
            image_ids.extend(iids)
            true_labels.extend([IDX2CLASS[l.item()] for l in labels])
        probs_np = np.vstack(probs_list)

    records = []
    for i in range(len(image_ids)):
        pred_idx = int(probs_np[i].argmax())
        row = {
            "image_id": image_ids[i],
            "true_label": true_labels[i],
            "pred_label": IDX2CLASS[pred_idx],
            "confidence": float(probs_np[i, pred_idx]),
        }
        for j, c in enumerate(CLASSES):
            row[f"p_{c}"] = float(probs_np[i, j])
        records.append(row)

    df = pd.DataFrame(records)

    # 4. Save diagnosis_val.parquet
    parquet_path = os.path.join(OUTPUT_DIR, "diagnosis_val.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\n[OK] Saved {parquet_path} ({len(df):,} rows)")

    # 5. Compute metrics
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mel_recall = float(recall_score(y_true, y_pred, labels=["mel"],
                                    average=None, zero_division=0)[0])

    prob_cols = [f"p_{c}" for c in CLASSES]
    probs_all = df[prob_cols].values
    labels_idx = np.array([CLASS2IDX[l] for l in y_true])
    ece = float(expected_calibration_error(probs_all, labels_idx))

    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    per_class = {}
    for c in CLASSES:
        per_class[c] = {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
        }

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "melanoma_recall": round(mel_recall, 4),
        "ECE": round(ece, 4),
        "temperature": round(agent.temperature, 4),
        "per_class": per_class,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "diagnosis_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved {metrics_path}")

    # 6. Confusion matrix plot (normalized by row)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("HAM10000 Confusion Matrix (Row-Normalized)", fontsize=12, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved {cm_path}")

    # 7. Summary table
    print("\n" + "=" * 62)
    print("        DiagnosisAgent Evaluation Summary (Validation Set)    ")
    print("=" * 62)
    print(f"  Accuracy:           {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.1f}%)")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Melanoma Recall:    {mel_recall:.4f} ({mel_recall*100:.1f}%)")
    print(f"  ECE (Calibrated):   {ece:.4f}")
    print(f"  Optimal Temperature:{agent.temperature:.4f}")
    print("-" * 62)
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>9} {'F1-Score':>10} {'Support':>9}")
    print("-" * 62)
    for c in CLASSES:
        p = per_class[c]
        print(f"  {c:<8} {p['precision']:>10.3f} {p['recall']:>9.3f} {p['f1']:>10.3f} {p['support']:>9d}")
    print("=" * 62)


if __name__ == "__main__":
    main()
