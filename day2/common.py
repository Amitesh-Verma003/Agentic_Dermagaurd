"""
Shared helpers: datasets, transforms, model loading and TTA inference.
"""
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CLASS_TO_IDX, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD, find_image


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Transforms ────────────────────────────────────────────────────────
def train_tf(img_size: int):
    # Dermoscopy is rotation/flip invariant → full rotations + flips.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])


def eval_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Datasets ──────────────────────────────────────────────────────────
class ImageTableDataset(Dataset):
    """Rows of a metadata DataFrame → (tensor, label or -1, index)."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = "dx" in self.df.columns
        self.paths = [find_image(img_dir, i) for i in self.df["image_id"]]
        missing = [i for i, p in zip(self.df["image_id"], self.paths) if p is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        y = CLASS_TO_IDX[self.df.at[i, "dx"]] if self.has_label else -1
        return x, y, i


def load_table(data_dir: str, metadata: Optional[str] = None) -> pd.DataFrame:
    """Load metadata.csv; if absent, list the images folder (no metadata)."""
    csv = metadata or os.path.join(data_dir, "metadata.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if "image_id" not in df.columns:            # tolerate e.g. 'id' / 'image'
            for alt in ("id", "image", "imageid", "image_name"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "image_id"})
                    break
        df["image_id"] = df["image_id"].astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
        return df
    img_dir = os.path.join(data_dir, "images") if os.path.isdir(os.path.join(data_dir, "images")) else data_dir
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    return pd.DataFrame({"image_id": ids})


def image_dir_of(data_dir: str) -> str:
    sub = os.path.join(data_dir, "images")
    return sub if os.path.isdir(sub) else data_dir


# ── Models ────────────────────────────────────────────────────────────
def build_model(name: str, pretrained: bool, drop_rate: float = 0.3):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES,
                             drop_rate=drop_rate)


def load_checkpoint(path: str, device):
    """Loads either a Day-1 checkpoint (efficientnet_b0 @224) or a Day-2 one."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt.get("model_name", "efficientnet_b0")
    img_size = int(ckpt.get("img_size", 224))
    model = build_model(name, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, name, img_size


def _tta_views(x: torch.Tensor, level: int):
    """level 1 = none, 2 = +hflip, 4 = flips, 8 = full dihedral group."""
    views = [x]
    if level >= 2:
        views.append(torch.flip(x, dims=[3]))
    if level >= 4:
        views += [torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
    if level >= 8:
        xt = x.transpose(2, 3)
        views += [xt, torch.flip(xt, dims=[3]), torch.flip(xt, dims=[2]), torch.flip(xt, dims=[2, 3])]
    return views


@torch.no_grad()
def predict_probs(model, df: pd.DataFrame, img_dir: str, img_size: int, device,
                  tta: int = 8, batch_size: int = 32, num_workers: int = 2,
                  return_views: bool = False):
    """Returns softmax probabilities (N, C) averaged over TTA views.
    If return_views, also returns per-view probs (V, N, C) for stability checks."""
    ds = ImageTableDataset(df, img_dir, eval_tf(img_size))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out, views_all = [], []
    for x, _, _ in dl:
        x = x.to(device)
        vp = torch.stack([F.softmax(model(v), dim=1) for v in _tta_views(x, tta)])  # V,B,C
        out.append(vp.mean(0).cpu())
        if return_views:
            views_all.append(vp.cpu())
    probs = torch.cat(out).numpy()
    if return_views:
        return probs, torch.cat(views_all, dim=1).numpy()
    return probs
