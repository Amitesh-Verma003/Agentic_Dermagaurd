"""
Agentic DermaGuard – Day 1 Validation & Triage Runner
======================================================
Runs *both* the Image Quality Agent and the Diagnosis Agent on the
full validation split, combines outputs into a triage decision, and
saves a comprehensive ``day1_summary.json`` report.

Triage logic:
  - RE_IMAGE       → quality_status == "RE_IMAGE_RECOMMENDED"
  - REFER          → melanoma (mel) or akiec predicted with high confidence
  - AUTO_TRIAGE    → everything else that passes quality
"""

import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

# ──── project imports ────
from dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    get_dataloaders,
    val_transforms,
)
from quality_agent import assess_quality

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_ROOT       = "/Users/vyomvarshney2005/Desktop/health-training/Health/training"
CKPT_PATH       = "best_diagnosis_agent.pth"
REPORT_PATH     = "day1_summary.json"
REFER_CLASSES   = {"mel", "akiec"}          # high-risk → always REFER
REFER_CONF_THRESH = 0.0                      # any mel/akiec prediction → REFER
QUALITY_PASS_THRESH = 0.50


def load_diagnosis_model(ckpt_path: str, device: torch.device):
    """Load the best checkpoint into a timm EfficientNet-B0."""
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_macro_f1={ckpt.get('val_macro_f1', 0):.4f}")
    return model


def triage_decision(predicted_class: str, confidence: float, quality_status: str) -> str:
    """Combine quality + diagnosis into a triage bucket."""
    if quality_status == "RE_IMAGE_RECOMMENDED":
        return "RE_IMAGE"
    if predicted_class in REFER_CLASSES and confidence >= REFER_CONF_THRESH:
        return "REFER"
    return "AUTO_TRIAGE"


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────
def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[eval] Device: {device}")

    # ── Data ──
    _, val_loader, _ = get_dataloaders(DATA_ROOT)
    print(f"[eval] Validation batches: {len(val_loader)}")

    # ── Load trained model ──
    if not os.path.exists(CKPT_PATH):
        print(f"❌  Checkpoint not found: {CKPT_PATH}")
        print("   Run train_diagnosis.py first.")
        sys.exit(1)
    model = load_diagnosis_model(CKPT_PATH, device)

    # ── Collect image paths for quality agent ──
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from dataset import VAL_RATIO, RANDOM_SEED

    csv_path = os.path.join(DATA_ROOT, "metadata.csv")
    img_dir  = os.path.join(DATA_ROOT, "images")
    df = pd.read_csv(csv_path)

    _, val_df = train_test_split(
        df, test_size=VAL_RATIO, stratify=df["dx"], random_state=RANDOM_SEED
    )
    val_df = val_df.reset_index(drop=True)

    # Build image_id → file path map
    img_paths: Dict[str, str] = {}
    for _, row in val_df.iterrows():
        img_id = row["image_id"]
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_paths[img_id] = candidate
                break

    # ── Run diagnosis on val set ──
    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    # ── Run quality agent on val set ──
    print(f"[eval] Running quality agent on {len(val_df)} images ...")
    quality_results = []
    t0 = time.time()
    for idx, row in val_df.iterrows():
        img_id = row["image_id"]
        path = img_paths.get(img_id)
        if path:
            qr = assess_quality(path)
        else:
            from quality_agent import QualityResult
            qr = QualityResult(
                quality_score=0.0,
                quality_status="RE_IMAGE_RECOMMENDED",
                reasons=["Image file not found"],
            )
        quality_results.append(qr)

        if (len(quality_results)) % 500 == 0:
            print(f"    quality checked {len(quality_results)}/{len(val_df)}")

    quality_time = time.time() - t0
    print(f"[eval] Quality agent done in {quality_time:.1f}s")

    # ── Combine into triage decisions ──
    triage_counts: Counter = Counter()
    quality_status_counts: Counter = Counter()
    per_image_results: List[dict] = []

    for i in range(len(all_preds)):
        pred_cls = CLASS_NAMES[all_preds[i]]
        true_cls = CLASS_NAMES[all_labels[i]]
        conf = all_confs[i]
        qr = quality_results[i]

        decision = triage_decision(pred_cls, conf, qr.quality_status)
        triage_counts[decision] += 1
        quality_status_counts[qr.quality_status] += 1

        per_image_results.append({
            "image_id": val_df.iloc[i]["image_id"],
            "true_label": true_cls,
            "predicted_label": pred_cls,
            "confidence": round(conf, 4),
            "quality_score": round(qr.quality_score, 4),
            "quality_status": qr.quality_status,
            "quality_reasons": qr.reasons,
            "triage_decision": decision,
        })

    # ── Compute metrics ──
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = {}
    for cls in CLASS_NAMES:
        if cls in report:
            per_class_metrics[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1-score":  round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }

    # ── Build summary ──
    summary = {
        "experiment": "Agentic DermaGuard – Day 1 Prototype",
        "model": "efficientnet_b0 (timm, pretrained)",
        "device": str(device),
        "validation_size": len(all_labels),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        },
        "per_class_metrics": per_class_metrics,
        "triage_distribution": dict(triage_counts),
        "quality_status_distribution": dict(quality_status_counts),
        "quality_agent_time_sec": round(quality_time, 1),
    }

    # ── Print ──
    print("\n" + "=" * 60)
    print("   AGENTIC DERMAGUARD – DAY 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Validation images : {summary['validation_size']}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Balanced Accuracy : {balanced_acc:.4f}")
    print()
    print("  Per-class metrics:")
    header = f"  {'Class':>8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for cls in CLASS_NAMES:
        m = per_class_metrics.get(cls, {})
        print(f"  {cls:>8s}  {m.get('precision',0):6.4f}  {m.get('recall',0):6.4f}  "
              f"{m.get('f1-score',0):6.4f}  {m.get('support',0):7d}")
    print()
    print("  Triage distribution:")
    for k, v in sorted(triage_counts.items()):
        print(f"    {k:16s}: {v:5d}  ({v/len(all_labels)*100:5.1f}%)")
    print()
    print("  Quality status distribution:")
    for k, v in sorted(quality_status_counts.items()):
        print(f"    {k:24s}: {v:5d}  ({v/len(all_labels)*100:5.1f}%)")
    print("=" * 60)

    # ── Save ──
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval] Full report saved → {REPORT_PATH}")

    # Also save per-image results for debugging
    per_image_path = "day1_per_image.json"
    with open(per_image_path, "w") as f:
        json.dump(per_image_results, f, indent=2)
    print(f"[eval] Per-image results saved → {per_image_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

