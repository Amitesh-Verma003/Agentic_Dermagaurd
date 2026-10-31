"""
Agentic DermaGuard – Image Quality Agent
=========================================
Fast rule-based quality checks using OpenCV:
  1. Blur detection       – Laplacian variance
  2. Poor illumination    – HSV V-channel mean
  3. Flash glare          – high V + low S area ratio
  4. Hair / obstruction   – morphological Black-Hat filter ratio

Returns quality_score (0–1), quality_status, and human-readable reasons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────
# Thresholds (tuned for dermatoscopic images)
# ──────────────────────────────────────────────
BLUR_LAPLACIAN_THRESH     = 50.0     # variance below → blurry
DARK_V_MEAN_THRESH        = 55.0     # V-channel mean below → too dark
BRIGHT_V_MEAN_THRESH      = 230.0    # V-channel mean above → too bright
GLARE_V_THRESH            = 245      # per-pixel V above → candidate glare
GLARE_S_THRESH            = 30       # per-pixel S below  → confirms glare
GLARE_AREA_RATIO_THRESH   = 0.02     # >2% of image area → glare problem
HAIR_BLACKHAT_RATIO_THRESH = 0.04    # >4% of image area → hair obstruction
MORPH_KERNEL_SIZE          = 17      # kernel for Black-Hat (odd, ≥15)


@dataclass
class QualityResult:
    """Container for per-image quality assessment."""
    quality_score:  float = 1.0
    quality_status: str = "PASS"
    reasons:        List[str] = field(default_factory=list)
    detail:         Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "quality_score":  round(self.quality_score, 4),
            "quality_status": self.quality_status,
            "reasons":        self.reasons,
            "detail":         {k: round(v, 4) for k, v in self.detail.items()},
        }


# ──────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────

def _check_blur(gray: np.ndarray, result: QualityResult) -> float:
    """Laplacian variance – lower = blurrier."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result.detail["blur_laplacian_var"] = lap_var
    if lap_var < BLUR_LAPLACIAN_THRESH:
        result.reasons.append(
            f"Image is blurry (Laplacian variance={lap_var:.1f}, "
            f"threshold={BLUR_LAPLACIAN_THRESH})"
        )
        # Penalty proportional to how far below the threshold
        return max(0.0, lap_var / BLUR_LAPLACIAN_THRESH)
    return 1.0


def _check_illumination(v_channel: np.ndarray, result: QualityResult) -> float:
    """Mean brightness of the V channel in HSV."""
    v_mean = float(v_channel.mean())
    result.detail["v_channel_mean"] = v_mean
    if v_mean < DARK_V_MEAN_THRESH:
        result.reasons.append(
            f"Image is too dark (V mean={v_mean:.1f}, "
            f"threshold={DARK_V_MEAN_THRESH})"
        )
        return max(0.0, v_mean / DARK_V_MEAN_THRESH)
    if v_mean > BRIGHT_V_MEAN_THRESH:
        result.reasons.append(
            f"Image is over-exposed (V mean={v_mean:.1f}, "
            f"threshold={BRIGHT_V_MEAN_THRESH})"
        )
        return max(0.0, 1.0 - (v_mean - BRIGHT_V_MEAN_THRESH) / (255 - BRIGHT_V_MEAN_THRESH))
    return 1.0


def _check_glare(hsv: np.ndarray, result: QualityResult) -> float:
    """Detect flash / specular glare: high V AND low S pixels."""
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    glare_mask = (v_channel > GLARE_V_THRESH) & (s_channel < GLARE_S_THRESH)
    glare_ratio = float(glare_mask.sum()) / glare_mask.size
    result.detail["glare_area_ratio"] = glare_ratio
    if glare_ratio > GLARE_AREA_RATIO_THRESH:
        result.reasons.append(
            f"Flash glare detected ({glare_ratio*100:.1f}% of image area, "
            f"threshold={GLARE_AREA_RATIO_THRESH*100:.1f}%)"
        )
        return max(0.0, 1.0 - (glare_ratio - GLARE_AREA_RATIO_THRESH) / 0.10)
    return 1.0


def _check_hair_obstruction(gray: np.ndarray, result: QualityResult) -> float:
    """Black-Hat morphology to highlight thin dark structures (hairs)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Threshold the blackhat image
    _, bh_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    hair_ratio = float(bh_mask.sum() / 255) / bh_mask.size
    result.detail["hair_obstruction_ratio"] = hair_ratio
    if hair_ratio > HAIR_BLACKHAT_RATIO_THRESH:
        result.reasons.append(
            f"Hair / obstruction detected ({hair_ratio*100:.1f}% of image, "
            f"threshold={HAIR_BLACKHAT_RATIO_THRESH*100:.1f}%)"
        )
        return max(0.0, 1.0 - (hair_ratio - HAIR_BLACKHAT_RATIO_THRESH) / 0.15)
    return 1.0


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def assess_quality(image_path: str) -> QualityResult:
    """Run all quality checks on a single image and return a QualityResult."""
    result = QualityResult()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        result.quality_score = 0.0
        result.quality_status = "RE_IMAGE_RECOMMENDED"
        result.reasons.append(f"Could not read image: {image_path}")
        return result

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    # Run all four checks – each returns a 0-1 sub-score
    scores = [
        _check_blur(gray, result),
        _check_illumination(v_ch, result),
        _check_glare(hsv, result),
        _check_hair_obstruction(gray, result),
    ]

    # Overall score: geometric mean (penalizes any single bad check)
    result.quality_score = float(np.prod(scores) ** (1.0 / len(scores)))
    result.quality_score = max(0.0, min(1.0, result.quality_score))

    if result.quality_score < 0.50 or len(result.reasons) >= 2:
        result.quality_status = "RE_IMAGE_RECOMMENDED"
    else:
        result.quality_status = "PASS"

    return result


def assess_quality_from_array(img_bgr: np.ndarray) -> QualityResult:
    """Same as assess_quality but accepts an already-loaded BGR array."""
    result = QualityResult()
    if img_bgr is None or img_bgr.size == 0:
        result.quality_score = 0.0
        result.quality_status = "RE_IMAGE_RECOMMENDED"
        result.reasons.append("Empty or None image array")
        return result

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    scores = [
        _check_blur(gray, result),
        _check_illumination(v_ch, result),
        _check_glare(hsv, result),
        _check_hair_obstruction(gray, result),
    ]

    result.quality_score = float(np.prod(scores) ** (1.0 / len(scores)))
    result.quality_score = max(0.0, min(1.0, result.quality_score))

    if result.quality_score < 0.50 or len(result.reasons) >= 2:
        result.quality_status = "RE_IMAGE_RECOMMENDED"
    else:
        result.quality_status = "PASS"

    return result


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import glob
    import os

    img_dir = "/Users/vyomvarshney2005/Desktop/health-training/Health/training/images"
    sample_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:5]

    if not sample_images:
        print("No sample images found.")
        sys.exit(1)

    for p in sample_images:
        res = assess_quality(p)
        print(f"{os.path.basename(p):30s}  "
              f"score={res.quality_score:.3f}  "
              f"status={res.quality_status:20s}  "
              f"reasons={res.reasons}")

