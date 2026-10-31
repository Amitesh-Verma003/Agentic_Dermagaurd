"""
FAST submission (no calibration needed).
  python3 day2/quick_submit.py --data "/path/to/hidden_folder"
Writes day2/outputs/hidden_submission.csv  (image_id,prediction,decision)
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from common import load_table, image_dir_of, load_checkpoint, predict_probs, select_device
from agents import quality as A1

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--metadata", default=None)
ap.add_argument("--out", default=str(C.OUTPUT_DIR / "hidden_submission.csv"))
a = ap.parse_args()

df = load_table(a.data, a.metadata)
img_dir = image_dir_of(a.data)
df = df[[C.find_image(img_dir, i) is not None for i in df.image_id]].reset_index(drop=True)
print(f"{len(df)} images in {img_dir}")

dev = select_device()
ckpts = [p for p in [str(C.ARTIFACT_DIR / "efficientnet_b0_256.pth"), str(C.DAY1_CKPT)] if os.path.exists(p)]
member_probs = []
for ck in ckpts:
    m, name, img = load_checkpoint(ck, dev)
    p = predict_probs(m, df, img_dir, img, dev, tta=8, num_workers=0)
    member_probs.append(p)
    print(f"  {os.path.basename(ck)} ({name}@{img}) done")
logp = np.mean([np.log(np.clip(p, 1e-7, 1)) for p in member_probs], axis=0)
probs = np.exp(logp - logp.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
pred = probs.argmax(1)
agree = np.all([p.argmax(1) == pred for p in member_probs], axis=0)

rows = []
for i, iid in enumerate(df.image_id):
    q = A1.assess(C.find_image(img_dir, iid))
    lab = C.CLASS_NAMES[pred[i]]
    conf = probs[i, pred[i]]
    pmal = probs[i, C.MALIGNANT_IDX].sum()
    if q["status"] == "FAIL":
        dec = "RE_IMAGE"
    elif lab in C.MALIGNANT or conf < 0.80 or pmal >= 0.15 or not agree[i]:
        dec = "REFER"
    else:
        dec = "AUTO_TRIAGE"
    rows.append((iid, lab, dec))

sub = pd.DataFrame(rows, columns=["image_id", "prediction", "decision"])
assert sub.image_id.is_unique
sub.to_csv(a.out, index=False)
print(f"\nSaved {a.out}")
print(sub.decision.value_counts().to_string())
print(sub.prediction.value_counts().to_string())
