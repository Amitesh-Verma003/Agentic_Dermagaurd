"""
Agentic DermaGuard – Dataset Loader
====================================
Handles two dataset layouts:
  A) Class-subfolders  → torchvision.datasets.ImageFolder
  B) Single images/ folder + metadata.csv  → custom Dataset

Optimized for Apple Silicon M1 with MPS backend.
"""

import os
import pathlib
from typing import Tuple, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Constants – M1-optimized DataLoader config
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = False          # Must be False for MPS / Unified Memory
IMG_SIZE = 224
VAL_RATIO = 0.20            # 80/20 stratified split when no val dir exists
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# HAM10000 class list (alphabetical → deterministic label encoding)
# ──────────────────────────────────────────────
CLASS_NAMES: List[str] = sorted(
    ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────
# Custom Dataset (CSV + flat image folder)
# ──────────────────────────────────────────────
class DermaCSVDataset(Dataset):
    """Loads images from a flat folder using a CSV that maps
    ``image_id`` → ``dx`` (diagnosis label)."""

    def __init__(
        self,
        image_dir: str,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_dir = pathlib.Path(image_dir)
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        # Detect image extension once
        sample_files = list(self.image_dir.iterdir())
        self.ext = sample_files[0].suffix if sample_files else ".jpg"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        label_str = row["dx"]
        label = CLASS_TO_IDX[label_str]

        # Try common extensions
        img_path = self.image_dir / f"{img_id}{self.ext}"
        if not img_path.exists():
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = self.image_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ──────────────────────────────────────────────
# Auto-detect layout & build DataLoaders
# ──────────────────────────────────────────────
def _detect_layout(root: str) -> str:
    """Return 'imagefolder' or 'csv' depending on directory structure."""
    root = pathlib.Path(root)
    csv_path = root / "metadata.csv"
    img_dir  = root / "images"

    # Layout B: flat images/ + metadata.csv
    if csv_path.exists() and img_dir.is_dir():
        return "csv"

    # Layout A: subdirectories named after classes
    subdirs = [d.name for d in root.iterdir() if d.is_dir()]
    if any(cls in subdirs for cls in CLASS_NAMES):
        return "imagefolder"

    raise FileNotFoundError(
        f"Cannot auto-detect dataset layout under {root}. "
        "Expected either class-subfolders OR images/ + metadata.csv."
    )


def get_dataloaders(
    data_root: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_ratio: float = VAL_RATIO,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Returns
    -------
    train_loader, val_loader, train_label_counts
        ``train_label_counts`` is a list of length NUM_CLASSES with the
        count for each class (used for inverse-frequency weighting).
    """
    layout = _detect_layout(data_root)
    print(f"[dataset] Detected layout: {layout}")

    if layout == "csv":
        csv_path = os.path.join(data_root, "metadata.csv")
        img_dir  = os.path.join(data_root, "images")
        df = pd.read_csv(csv_path)

        # Stratified train / val split
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            stratify=df["dx"],
            random_state=RANDOM_SEED,
        )
        print(f"[dataset] Train: {len(train_df)}  |  Val: {len(val_df)}")

        train_ds = DermaCSVDataset(img_dir, train_df, transform=train_transforms)
        val_ds   = DermaCSVDataset(img_dir, val_df,   transform=val_transforms)

        # Class counts from training split
        train_label_counts = [0] * NUM_CLASSES
        for _, row in train_df.iterrows():
            train_label_counts[CLASS_TO_IDX[row["dx"]]] += 1

    else:  # imagefolder
        train_ds = datasets.ImageFolder(data_root, transform=train_transforms)
        val_ds   = datasets.ImageFolder(data_root, transform=val_transforms)

        # Override class_to_idx to match our canonical order
        train_ds.class_to_idx = CLASS_TO_IDX
        val_ds.class_to_idx   = CLASS_TO_IDX

        train_label_counts = [0] * NUM_CLASSES
        for _, label in train_ds.samples:
            train_label_counts[label] += 1

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print(f"[dataset] Class counts (train): {dict(zip(CLASS_NAMES, train_label_counts))}")
    return train_loader, val_loader, train_label_counts


# ──────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    DATA_ROOT = "/Users/vyomvarshney2005/Desktop/health-training/Health/training"

    train_loader, val_loader, counts = get_dataloaders(DATA_ROOT)

    # Use num_workers=0 loader for sanity check to avoid shared-memory issues
    # in restricted environments. The full pipeline uses num_workers=2.
    from torch.utils.data import DataLoader as DL
    sanity_loader = DL(train_loader.dataset, batch_size=4, shuffle=False,
                       num_workers=0, pin_memory=False)
    imgs, labels = next(iter(sanity_loader))
    print(f"[sanity] batch shape : {imgs.shape}")
    print(f"[sanity] label batch : {labels[:8]}")
    print(f"[sanity] device ready: mps={torch.backends.mps.is_available()}")
