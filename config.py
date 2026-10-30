"""
config.py – Single source of truth for all hyperparameters and paths.
"""
import os
import random
import numpy as np
import torch


# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_CSV   = r"C:\Users\dubey\OneDrive\Desktop\health-training\Health\training\metadata.csv"
TRAIN_DIR   = r"C:\Users\dubey\OneDrive\Desktop\health-training\Health\training\images"
VAL_CSV     = r"C:\Users\dubey\OneDrive\Desktop\health-eval (1)\Health\eval\metadata.csv"
VAL_DIR     = r"C:\Users\dubey\OneDrive\Desktop\health-eval (1)\Health\eval\images"
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")

# ── Label info ─────────────────────────────────────────────────────────────
CLASSES     = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS   = {i: c for i, c in enumerate(CLASSES)}

# CSV column names
COL_IMAGE_ID = "image_id"
COL_LABEL    = "dx"

# ── Model / training ──────────────────────────────────────────────────────
MODEL_NAME     = "efficientnet_b0"       # timm model name
IMG_SIZE       = 224
BATCH_SIZE     = 16                      # safe for 4 GB VRAM
NUM_WORKERS    = 2
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
EPOCHS         = 12
WARMUP_EPOCHS  = 1
LABEL_SMOOTHING = 0.1
USE_AMP        = True                    # mixed‐precision

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── ImageNet normalization ─────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Checkpoint / calibration paths ─────────────────────────────────────────
BEST_CHECKPOINT = os.path.join(OUTPUT_DIR, "best.pt")
TEMPERATURE_PATH = os.path.join(OUTPUT_DIR, "temperature.json")


def seed_everything(seed: int = SEED):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
