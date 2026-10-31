"""
Inference script on external evaluation dataset:
/Users/vyomvarshney2005/Desktop/health-eval (1)/Health/eval
Runs:
1. Diagnosis Agent (best_diagnosis_agent.pth)
2. Image Quality Agent (quality_agent.py)
Outputs exact numerical counts:
- Total images
- Correctly predicted images & accuracy
- RE_IMAGE count & PASS count
- Triage breakdown
"""

import json
import os
import sys
import time
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import timm
from quality_agent import assess_quality
from dataset import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES, DermaCSVDataset

EVAL_DIR = "/Users/vyomvarshney2005/Desktop/health-eval (1)/Health/eval"
CSV_PATH = os.path.join(EVAL_DIR, "metadata.csv")
IMAGES_DIR = os.path.join(EVAL_DIR, "images")
CKPT_PATH = "/Users/vyomvarshney2005/Desktop/health-training/best_diagnosis_agent.pth"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[inference] Running on device: {device}")

    # Load metadata
    df = pd.read_csv(CSV_PATH)
    total_samples = len(df)
    print(f"[inference] Found {total_samples} evaluation images in metadata.")

    # Dataset & Loader
    eval_dataset = DermaCSVDataset(IMAGES_DIR, df, transform=val_transforms)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues
        pin_memory=False,
    )

    # Load Diagnosis Model
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("[inference] Loaded diagnosis agent model successfully.")

    # Diagnosis Inference
    all_preds = []
    all_labels = []
    all_confs = []

    print("[inference] Running Diagnosis Agent...")
    t0 = time.time()
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
    diag_time = time.time() - t0
    print(f"[inference] Diagnosis inference completed in {diag_time:.2f}s")

    # Diagnosis evaluation
    correct_count = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    diag_accuracy = correct_count / total_samples

    # Quality Agent Inference
    print("[inference] Running Image Quality Agent...")
    t1 = time.time()
    quality_counts = Counter()
    quality_results = []
    for idx, row in df.iterrows():
        img_id = row["image_id"]
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            for ext in [".jpeg", ".png", ".bmp"]:
                cand = os.path.join(IMAGES_DIR, f"{img_id}{ext}")
                if os.path.exists(cand):
                    img_path = cand
                    break
        
        qr = assess_quality(img_path)
        quality_counts[qr.quality_status] += 1
        quality_results.append(qr)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_samples} images...")
    qual_time = time.time() - t1
    print(f"[inference] Image Quality Agent completed in {qual_time:.2f}s")

    # Per-class accuracy breakdown
    per_class_correct = Counter()
    per_class_total = Counter()
    for p, y in zip(all_preds, all_labels):
        cls_name = CLASS_NAMES[y]
        per_class_total[cls_name] += 1
        if p == y:
            per_class_correct[cls_name] += 1

    # Triage decision
    triage_counts = Counter()
    for pred_idx, qr in zip(all_preds, quality_results):
        pred_cls = CLASS_NAMES[pred_idx]
        if qr.quality_status == "RE_IMAGE_RECOMMENDED":
            triage_counts["RE_IMAGE"] += 1
        elif pred_cls in {"mel", "akiec"}:
            triage_counts["REFER"] += 1
        else:
            triage_counts["AUTO_TRIAGE"] += 1

    # Print results
    print("\n" + "=" * 60)
    print("      HEALTH-EVAL INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total Evaluation Images: {total_samples}")
    print(f"Correctly Predicted:     {correct_count} / {total_samples} ({diag_accuracy * 100:.2f}%)")
    print(f"Incorrectly Predicted:   {total_samples - correct_count} / {total_samples} ({(1 - diag_accuracy) * 100:.2f}%)")
    print("-" * 60)
    print("IMAGE QUALITY AGENT BREAKDOWN:")
    print(f"  PASS:                  {quality_counts['PASS']} ({quality_counts['PASS'] / total_samples * 100:.2f}%)")
    print(f"  RE_IMAGE_RECOMMENDED:  {quality_counts['RE_IMAGE_RECOMMENDED']} ({quality_counts['RE_IMAGE_RECOMMENDED'] / total_samples * 100:.2f}%)")
    print("-" * 60)
    print("TRIAGE ACTION BREAKDOWN:")
    for k, v in triage_counts.items():
        print(f"  {k:15s}: {v:5d} ({v / total_samples * 100:.2f}%)")
    print("-" * 60)
    print("PER-CLASS ACCURACY BREAKDOWN:")
    for cls in CLASS_NAMES:
        tot = per_class_total[cls]
        cor = per_class_correct[cls]
        acc_pct = (cor / tot * 100) if tot > 0 else 0
        print(f"  {cls:8s}: {cor:4d} / {tot:4d} ({acc_pct:5.2f}%)")
    print("=" * 60)

    # Save results to json
    results = {
        "dataset_path": EVAL_DIR,
        "total_images": total_samples,
        "correctly_predicted": correct_count,
        "diagnosis_accuracy": round(diag_accuracy, 4),
        "quality_re_image_count": quality_counts["RE_IMAGE_RECOMMENDED"],
        "quality_pass_count": quality_counts["PASS"],
        "triage_distribution": dict(triage_counts),
        "per_class_performance": {
            cls: {
                "correct": per_class_correct[cls],
                "total": per_class_total[cls],
                "accuracy": round((per_class_correct[cls] / per_class_total[cls]) if per_class_total[cls] > 0 else 0, 4)
            } for cls in CLASS_NAMES
        }
    }

    out_file = "/Users/vyomvarshney2005/Desktop/health-training/health_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[inference] Saved numerical results to {out_file}")

if __name__ == "__main__":
    main()

