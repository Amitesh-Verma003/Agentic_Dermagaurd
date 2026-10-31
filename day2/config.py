"""
Agentic DermaGuard – shared configuration (Day 2)
=================================================
Edit the paths below if your folders live somewhere else.
Every script also accepts command-line overrides.
"""
import os
import pathlib

# ── Folders ────────────────────────────────────────────────────────────
DAY2_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = DAY2_DIR.parent                                  # health-training/

TRAIN_DIR = os.environ.get("DG_TRAIN_DIR", str(ROOT_DIR / "Health" / "training"))
# The eval (validation) pack you used on Day 1:
EVAL_DIR = os.environ.get(
    "DG_EVAL_DIR", "/Users/vyomvarshney2005/Desktop/health-eval (1)/Health/eval"
)

ARTIFACT_DIR = DAY2_DIR / "artifacts"      # checkpoints, calibration, cached preds
OUTPUT_DIR = DAY2_DIR / "outputs"          # CSVs, reports, per-image JSON
ARTIFACT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DAY1_CKPT = ROOT_DIR / "best_diagnosis_agent.pth"           # efficientnet_b0 from Day 1

# ── Classes ───────────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # alphabetical, same as Day 1
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

CLASS_FULL_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
# Malignant or pre-malignant → never auto-triaged
MALIGNANT = {"mel", "bcc", "akiec"}
MALIGNANT_IDX = [CLASS_TO_IDX[c] for c in sorted(MALIGNANT)]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_image(img_dir, image_id):
    """Return the path of image_id inside img_dir (any common extension)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
        p = os.path.join(img_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return None
