"""
dataset.py – PyTorch Dataset for HAM10000 skin lesion images.

Returns (image_tensor, label_idx, image_id) per sample.
Filters out CSV rows whose image file is missing on disk.
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    CLASSES, CLASS2IDX, COL_IMAGE_ID, COL_LABEL,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


def _build_df(csv_path: str, img_dir: str) -> pd.DataFrame:
    """Load metadata CSV and keep only rows whose .jpg exists."""
    df = pd.read_csv(csv_path)
    df["filepath"] = df[COL_IMAGE_ID].apply(
        lambda x: os.path.join(img_dir, f"{x}.jpg")
    )
    before = len(df)
    df = df[df["filepath"].apply(os.path.isfile)].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  [!] Filtered {before - after} rows with missing images "
              f"({before} -> {after})")
    return df


# ── Augmentations ─────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset ───────────────────────────────────────────────────────────────

class SkinLesionDataset(Dataset):
    """HAM10000 skin‐lesion dataset.

    Parameters
    ----------
    csv_path : path to metadata.csv
    img_dir  : folder containing ISIC_*.jpg images
    transform: torchvision transform pipeline
    """

    def __init__(self, csv_path: str, img_dir: str,
                 transform=None):
        self.df = _build_df(csv_path, img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CLASS2IDX[row[COL_LABEL]]
        return img, label, row[COL_IMAGE_ID]

    def class_counts(self) -> dict:
        """Return {class_name: count} from the filtered dataframe."""
        counts = self.df[COL_LABEL].value_counts().to_dict()
        return {c: counts.get(c, 0) for c in CLASSES}
