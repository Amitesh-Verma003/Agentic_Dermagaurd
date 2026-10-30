"""
agents/diagnosis_agent.py – Inference wrapper for the DiagnosisAgent.

Usage:
    agent = DiagnosisAgent("outputs/best.pt", "outputs/temperature.json")
    result = agent.predict("path/to/image.jpg")
    # -> {"label": "mel", "confidence": 0.87, "probabilities": {...}}
"""
import json
from typing import Union

import numpy as np
import torch
from PIL import Image
import timm

from config import (
    BEST_CHECKPOINT, CLASSES, IDX2CLASS, IMG_SIZE, MODEL_NAME,
    NUM_CLASSES, TEMPERATURE_PATH, seed_everything, SEED,
)
from dataset import val_transforms


class DiagnosisAgent:
    """Skin-lesion diagnosis agent backed by a calibrated EfficientNet-B0."""

    def __init__(self,
                 checkpoint_path: str = BEST_CHECKPOINT,
                 temperature_path: str = TEMPERATURE_PATH):
        seed_everything(SEED)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model
        self.model = timm.create_model(MODEL_NAME, pretrained=False,
                                       num_classes=NUM_CLASSES)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        # Load temperature
        self.temperature = 1.0
        try:
            with open(temperature_path) as f:
                self.temperature = json.load(f)["temperature"]
        except FileNotFoundError:
            print("[!] temperature.json not found -- using T=1.0")

        self.transform = val_transforms

    # ── Single prediction ─────────────────────────────────────────────

    def predict(self, image_path_or_pil: Union[str, Image.Image]) -> dict:
        """Predict a single image.

        Parameters
        ----------
        image_path_or_pil : str or PIL.Image.Image

        Returns
        -------
        dict with keys: label, confidence, probabilities
        """
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits / self.temperature, dim=1)

        probs_np = probs.cpu().numpy()[0]
        pred_idx = int(probs_np.argmax())

        return {
            "label": IDX2CLASS[pred_idx],
            "confidence": round(float(probs_np[pred_idx]), 4),
            "probabilities": {
                c: round(float(probs_np[i]), 4) for i, c in enumerate(CLASSES)
            },
        }

    # ── Batch prediction ──────────────────────────────────────────────

    def predict_batch(self, image_paths: list) -> list:
        """Predict a list of images.

        Parameters
        ----------
        image_paths : list of str (file paths) or PIL.Image.Image

        Returns
        -------
        list of dicts (same format as predict)
        """
        tensors = []
        for p in image_paths:
            if isinstance(p, str):
                img = Image.open(p).convert("RGB")
            else:
                img = p.convert("RGB")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits / self.temperature, dim=1)

        probs_np = probs.cpu().numpy()
        results = []
        for i in range(len(image_paths)):
            pred_idx = int(probs_np[i].argmax())
            results.append({
                "label": IDX2CLASS[pred_idx],
                "confidence": round(float(probs_np[i, pred_idx]), 4),
                "probabilities": {
                    c: round(float(probs_np[i, j]), 4)
                    for j, c in enumerate(CLASSES)
                },
            })
        return results
