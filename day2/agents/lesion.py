"""
Agent 3 – Lesion Analysis Agent (dermoscopic ABC(D) cues)
=========================================================
Independent of the CNN: classical image analysis that a panel can inspect.

  1. Hair removal (black-hat + inpainting), vignette masking.
  2. Lesion segmentation: Otsu on the blurred Lab lightness inside the dermoscope
     field, keep the largest central component, fill holes.
  3. Features
       A  asymmetry   – mask overlap after flipping about both principal axes
       B  border      – compactness P²/4πA, solidity A/convex-hull, border sharpness
       C  colour      – number of dermoscopic colours present (white, red,
                        light-brown, dark-brown, blue-grey, black) + Lab spread
       D  size        – equivalent diameter relative to the field (no mm scale)
  4. Suspicion score = P(malignant | ABC features) from a logistic regression fitted
     on the training images (calibrate.py → artifacts/lesion_model.json).
     Falls back to a TDS-style weighted sum if no model is fitted.
  5. Reliability – how far the segmentation can be trusted (lesion size,
     border contact, lesion/skin contrast, image quality).
"""
import json
import math
import os
from typing import Optional

import cv2
import numpy as np

FEATURES = ["asym_mean", "asym_max", "compactness", "solidity", "border_sharpness",
            "n_colours", "L_std", "a_std", "b_std", "rel_diameter", "contrast",
            "c_white", "c_red", "c_lightbrown", "c_darkbrown", "c_bluegrey", "c_black"]


def _remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, m = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    if m.mean() / 255 > 0.002:
        img = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)
    return img


def _field(v):
    dark = (v < 35).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return ~np.isin(lab, [b for b in border if b != 0])


def _flat_field(L, field):
    """Remove dermoscope vignetting / uneven light: fit a quadratic surface to the
    brighter (skin) pixels and normalise by it, so only the lesion stays dark."""
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xn, yn = (xx - w / 2) / w, (yy - h / 2) / h
    A = np.stack([np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn], -1).reshape(-1, 6)
    sel = field.ravel() & (L.ravel() >= np.percentile(L[field], 40) if field.sum() > 500 else True)
    if np.sum(sel) < 100:
        return L
    coef, *_ = np.linalg.lstsq(A[sel], L.ravel()[sel], rcond=None)
    bg = np.clip((A @ coef).reshape(h, w), 20, 255)
    return np.clip(L / bg * 200.0, 0, 255).astype(np.float32)


def segment(img_bgr):
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (256, int(round(256 * h / w))), interpolation=cv2.INTER_AREA)
    img = _remove_hair(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    field = _field(hsv[..., 2])
    L = cv2.GaussianBlur(lab[..., 0], (7, 7), 0).astype(np.float32)
    L = _flat_field(L, field)
    field = field & (cv2.erode(field.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    vals = L[field] if field.sum() > 500 else L.ravel()
    thr, _ = cv2.threshold(np.clip(vals, 0, 255).reshape(-1, 1).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((L < thr) & field).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return img, lab, field, None
    hh, ww = mask.shape
    best, best_s = None, -1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        d = math.hypot((cx - ww / 2) / ww, (cy - hh / 2) / hh)
        comp = labels == i
        touch = np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]).mean()
        if d > 0.38 or touch > 0.15 or area < 0.003 * hh * ww:
            continue                      # corner vignette / frame edge / speck – not the lesion
        s = area * (1.2 - d) * (0.25 if touch > 0.05 else 1.0)
        if s > best_s:
            best, best_s = i, s
    if best is None:
        return img, lab, field, None
    m = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 1, thickness=-1)       # fill holes
    return img, lab, field, m.astype(bool)


def _asymmetry(mask):
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)
    ang = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
    h, w = mask.shape
    side = int(2 * max(h, w))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = side // 2 - int(cy), side // 2 - int(cx)
    canvas[oy:oy + h, ox:ox + w] = mask.astype(np.uint8)
    R = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
    rot = cv2.warpAffine(canvas, R, (side, side), flags=cv2.INTER_NEAREST) > 0
    area = rot.sum()
    a1 = np.logical_xor(rot, rot[::-1, :]).sum() / (2 * area)
    a2 = np.logical_xor(rot, rot[:, ::-1]).sum() / (2 * area)
    return float((a1 + a2) / 2), float(max(a1, a2))


def _colours(lab, mask):
    L = lab[..., 0].astype(float)[mask] * 100 / 255
    a = lab[..., 1].astype(float)[mask] - 128
    b = lab[..., 2].astype(float)[mask] - 128
    chroma = np.hypot(a, b)
    n = max(1, mask.sum())
    frac = {
        "c_white": ((L > 80) & (chroma < 12)).sum() / n,
        "c_red": ((a > 25) & (L > 30) & (L < 80)).sum() / n,
        "c_lightbrown": ((L >= 45) & (L <= 80) & (b > 12) & (a > 5) & (a <= 25)).sum() / n,
        "c_darkbrown": ((L >= 20) & (L < 45) & (b > 6) & (a > 3)).sum() / n,
        "c_bluegrey": ((b < 2) & (L >= 25) & (L <= 70) & (a < 15)).sum() / n,
        "c_black": (L < 20).sum() / n,
    }
    return {k: float(v) for k, v in frac.items()}


def extract(img_bgr: np.ndarray):
    img, lab, field, mask = segment(img_bgr)
    if mask is None or mask.sum() < 50:
        return None, None
    area = mask.sum()
    fa = max(1, field.sum())
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    solidity = float(area / max(1.0, cv2.contourArea(hull)))
    compact = float(per ** 2 / (4 * math.pi * area))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    sharp = float(np.hypot(gx, gy)[edge].mean()) if edge.any() else 0.0
    am, ax = _asymmetry(mask)
    col = _colours(lab, mask)
    Lch = lab[..., 0].astype(float)
    skin = field & ~cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    contrast = float(Lch[skin].mean() - Lch[mask].mean()) if skin.sum() > 100 else 0.0
    h, w = mask.shape
    border_touch = float(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean())
    f = {
        "asym_mean": am, "asym_max": ax, "compactness": compact, "solidity": solidity,
        "border_sharpness": sharp, "n_colours": float(sum(v > 0.05 for v in col.values())),
        "L_std": float(Lch[mask].std()), "a_std": float(lab[..., 1][mask].std()),
        "b_std": float(lab[..., 2][mask].std()),
        "rel_diameter": float(2 * math.sqrt(area / math.pi) / math.sqrt(fa)),
        "contrast": contrast, **col,
    }
    seg = {"area_frac": float(area / fa), "border_touch": border_touch, "mask_shape": [h, w]}
    return f, seg


class LesionAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            with open(model_path) as fh:
                self.model = json.load(fh)

    def suspicion(self, f):
        if self.model:
            x = np.array([f[k] for k in self.model["features"]], float)
            z = (x - np.array(self.model["mean"])) / np.array(self.model["std"])
            logit = float(np.dot(z, self.model["coef"]) + self.model["intercept"])
            return 1 / (1 + math.exp(-logit)), "logistic model (fitted on training images)"
        # TDS-like fallback: A*1.3 + B*0.1 + C*0.5 (+ structure proxy), squashed to 0–1
        A = 2 * min(1, f["asym_mean"] / 0.15)
        B = 8 * min(1, max(0, (f["compactness"] - 1.1) / 1.0))
        tds = 1.3 * A + 0.1 * B + 0.5 * f["n_colours"] + 0.5 * min(5, f["L_std"] / 6)
        return 1 / (1 + math.exp(-(tds - 4.75))), "TDS-style rule (no fitted model)"

    def run(self, image_path: Optional[str] = None, img_bgr: Optional[np.ndarray] = None,
            quality_score: float = 1.0, features: Optional[dict] = None, seg: Optional[dict] = None):
        if features is None and (image_path or img_bgr is not None):
            img_bgr = cv2.imread(image_path) if img_bgr is None else img_bgr
            features, seg = extract(img_bgr) if img_bgr is not None else (None, None)
        if features is None:
            return {"agent": "A3_lesion", "suspicion_score": None, "suspicion_level": "UNKNOWN",
                    "reliability": 0.0, "reliability_level": "LOW",
                    "evidence": ["Lesion could not be segmented – no ABC evidence"], "features": {}}

        p, how = self.suspicion(features)
        thr = (self.model or {}).get("levels", [0.35, 0.65])
        level = "HIGH" if p >= thr[1] else ("MODERATE" if p >= thr[0] else "LOW")

        # Reliability of the segmentation-based evidence
        rel = 1.0
        notes = []
        af = seg["area_frac"]
        if af < 0.02:
            rel *= 0.4; notes.append("lesion very small in frame")
        elif af > 0.80:
            rel *= 0.4; notes.append("lesion fills the frame – border not fully visible")
        if seg["border_touch"] > 0.25:
            rel *= 0.6; notes.append("lesion touches image edge")
        if features["contrast"] < 8:
            rel *= 0.5; notes.append("low lesion/skin contrast")
        rel *= 0.5 + 0.5 * quality_score
        rlevel = "HIGH" if rel >= 0.7 else ("MEDIUM" if rel >= 0.4 else "LOW")

        ev = []
        a = features["asym_mean"]
        ev.append(f"Asymmetry {a:.2f} – " + ("marked" if a > 0.20 else "mild" if a > 0.10 else "symmetric"))
        cp, so = features["compactness"], features["solidity"]
        ev.append(f"Border compactness {cp:.2f}, solidity {so:.2f} – " +
                  ("irregular" if (cp > 1.8 or so < 0.85) else "regular"))
        cols = [k[2:].replace("lightbrown", "light-brown").replace("darkbrown", "dark-brown")
                .replace("bluegrey", "blue-grey") for k in
                ["c_white", "c_red", "c_lightbrown", "c_darkbrown", "c_bluegrey", "c_black"]
                if features[k] > 0.05]
        ev.append(f"{len(cols)} dermoscopic colour(s): {', '.join(cols) if cols else 'uniform'}")
        if features["c_bluegrey"] > 0.05 or features["c_white"] > 0.05:
            ev.append("Blue-grey / white areas present (regression-type structures)")
        ev.append(f"Relative diameter {features['rel_diameter']:.2f} of field")
        if notes:
            ev.append("Reliability reduced: " + "; ".join(notes))

        return {"agent": "A3_lesion", "suspicion_score": round(p, 4), "suspicion_level": level,
                "method": how, "reliability": round(rel, 3), "reliability_level": rlevel,
                "evidence": ev,
                "features": {k: round(v, 4) for k, v in features.items()},
                "segmentation": {k: v for k, v in seg.items() if k != "mask_shape"}}
