"""
Agent 1 – Image Quality Agent (v2)
==================================
Rule-based checks, each with a stated reason and a data-justified threshold:

  check          measure                                         why
  ------------   ---------------------------------------------   ---------------------------------
  blur           variance of Laplacian on the central field,     out-of-focus images lose the fine
                 image rescaled to 300 px wide                    network / border detail the model needs
  illumination   mean V (HSV) inside the dermoscope field         under/over-exposed images hide colour cues
  glare          % of field with V>245 and S<30                   specular reflections mimic white/blue-grey
  obstruction    % of field covered by thin dark structures       dense hair / ruler marks cover the lesion
                 (morphological black-hat)
  field          % of frame that is dermoscope vignette (black)   lesion squeezed into a small usable area

Differences from Day 1:
  * Measurements ignore the black circular vignette of the dermoscope, which made
    normal images look "too dark".
  * Blur is measured at a fixed scale, so the threshold means the same thing for
    any image size.
  * Each check returns PASS / WARN / FAIL. Hair is common in dermoscopy, so it only
    WARNs unless it is severe.
  * Thresholds default to the values below; calibrate.py replaces them with
    percentiles from the 7,762 training images (FAIL ≈ worst 1%, WARN ≈ worst 5%)
    and stores them in artifacts/calibration.json.

status: FAIL (→ RE_IMAGE) if any check FAILs or ≥3 checks WARN, WARN if any WARN, else PASS.
"""
from typing import Dict, Optional

import cv2
import numpy as np

DEFAULT_THRESHOLDS = {
    # (warn, fail). For blur, lower is worse; for the others, higher is worse.
    "blur": [40.0, 15.0],
    "dark": [70.0, 45.0],           # field mean V below → too dark
    "bright": [235.0, 248.0],       # field mean V above → over-exposed
    "glare": [0.03, 0.10],
    "obstruction": [0.10, 0.22],
    "vignette": [0.45, 0.70],
}


def _field_mask(v: np.ndarray) -> np.ndarray:
    """Pixels inside the dermoscope field (removes black vignette connected to the border)."""
    dark = (v < 35).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    vign = np.isin(lab, [b for b in border if b != 0])
    return ~vign


def measure(img_bgr: np.ndarray) -> Dict[str, float]:
    h, w = img_bgr.shape[:2]
    scale = 300.0 / w
    small = cv2.resize(img_bgr, (300, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    field = _field_mask(v)
    if field.mean() < 0.05:            # entirely dark image → measure everything
        field = np.ones_like(field, dtype=bool)

    hh, ww = gray.shape
    centre = np.zeros_like(field)
    centre[hh // 6: 5 * hh // 6, ww // 6: 5 * ww // 6] = True
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    m = field & centre
    blur = float(lap[m].var()) if m.sum() > 100 else float(lap.var())

    glare = float(((v > 245) & (s < 30) & field).sum() / field.sum())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    obstruction = float(((bh > 25) & field).sum() / field.sum())
    return {
        "blur_laplacian_var": blur,
        "field_v_mean": float(v[field].mean()),
        "glare_ratio": glare,
        "obstruction_ratio": obstruction,
        "vignette_ratio": float(1.0 - field.mean()),
    }


def _grade(value, warn, fail, lower_is_worse=False):
    if lower_is_worse:
        if value < fail:
            return "FAIL", max(0.0, value / max(fail, 1e-6)) * 0.5
        if value < warn:
            return "WARN", 0.5 + 0.5 * (value - fail) / max(warn - fail, 1e-6)
        return "PASS", 1.0
    if value > fail:
        return "FAIL", 0.5 * max(0.0, 1 - (value - fail) / max(fail, 1e-6))
    if value > warn:
        return "WARN", 0.5 + 0.5 * (fail - value) / max(fail - warn, 1e-6)
    return "PASS", 1.0


def assess(image_path: Optional[str] = None, img_bgr: Optional[np.ndarray] = None,
           thresholds: Optional[dict] = None, metrics: Optional[dict] = None) -> dict:
    """Grade an image. Pass pre-computed `metrics` (from measure()) to skip re-reading."""
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    if metrics is None and img_bgr is None:
        img_bgr = cv2.imread(image_path) if image_path else None
    if metrics is None and (img_bgr is None or img_bgr.size == 0):
        return {"agent": "A1_quality", "status": "FAIL", "score": 0.0,
                "reasons": ["Image could not be read"], "checks": {}, "metrics": {}}

    mt = metrics if metrics is not None else measure(img_bgr)
    checks, reasons, subs = {}, [], []

    g, sc = _grade(mt["blur_laplacian_var"], *t["blur"], lower_is_worse=True)
    checks["blur"] = g; subs.append(sc)
    if g != "PASS":
        reasons.append(f"{'Blurry' if g == 'FAIL' else 'Slightly soft'} image "
                       f"(sharpness {mt['blur_laplacian_var']:.0f} < {t['blur'][0 if g == 'WARN' else 1]:.0f})")

    vm = mt["field_v_mean"]
    gd, sd = _grade(vm, *t["dark"], lower_is_worse=True)
    gb, sb = _grade(vm, *t["bright"])
    if gd != "PASS":
        checks["illumination"] = gd; subs.append(sd)
        reasons.append(f"Under-exposed (brightness {vm:.0f})")
    elif gb != "PASS":
        checks["illumination"] = gb; subs.append(sb)
        reasons.append(f"Over-exposed (brightness {vm:.0f})")
    else:
        checks["illumination"] = "PASS"; subs.append(1.0)

    for key, label, metric in (("glare", "Specular glare", "glare_ratio"),
                               ("obstruction", "Hair / marks obstruct the lesion", "obstruction_ratio"),
                               ("vignette", "Small usable field (dermoscope vignette)", "vignette_ratio")):
        g, sc = _grade(mt[metric], *t[key])
        checks[key] = g; subs.append(sc)
        if g != "PASS":
            reasons.append(f"{label}: {mt[metric]*100:.1f}% of image")

    score = float(np.prod(np.clip(subs, 1e-3, 1.0)) ** (1.0 / len(subs)))
    n_fail = sum(v == "FAIL" for v in checks.values())
    n_warn = sum(v == "WARN" for v in checks.values())
    status = "FAIL" if (n_fail or n_warn >= 3) else ("WARN" if n_warn else "PASS")
    if not reasons:
        reasons = ["Sharp, evenly lit, unobstructed – image is usable"]
    return {"agent": "A1_quality", "status": status, "score": round(score, 4),
            "reasons": reasons, "checks": checks,
            "metrics": {k: round(v, 4) for k, v in mt.items()}}


def thresholds_from_metrics(metric_rows) -> dict:
    """Data-justified thresholds from training-set measurements:
    FAIL ≈ worst 1 %, WARN ≈ worst 5 % (with sane floors/ceilings)."""
    import pandas as pd
    d = pd.DataFrame(metric_rows)
    q = lambda c, p: float(np.nanpercentile(d[c], p))
    return {
        "blur": [max(q("blur_laplacian_var", 5), 8.0), max(q("blur_laplacian_var", 1), 4.0)],
        "dark": [min(q("field_v_mean", 5), 90.0), min(q("field_v_mean", 1), 60.0)],
        "bright": [max(q("field_v_mean", 95), 225.0), max(q("field_v_mean", 99), 240.0)],
        "glare": [max(q("glare_ratio", 95), 0.02), max(q("glare_ratio", 99), 0.06)],
        "obstruction": [max(q("obstruction_ratio", 95), 0.08), max(q("obstruction_ratio", 99), 0.15)],
        "vignette": [max(q("vignette_ratio", 95), 0.40), max(q("vignette_ratio", 99), 0.60)],
    }
