"""
Six-agent pipeline core – used by run_pipeline.py, demo.py and calibrate.py.
"""
import json
import multiprocessing as mp
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd

import config as C
from agents import quality as A1
from agents import lesion as A3
from agents.context import ContextAgent
from agents.lesion import LesionAgent
from agents.coordinator import Coordinator, fuse_arrays
from agents.diagnosis import DiagnosisAgent

CALIB_PATH = C.ARTIFACT_DIR / "calibration.json"


# ── per-image classical features (A1 measurements + A3 ABC features) ──────
def image_features(path):
    im = cv2.imread(path) if path else None
    if im is None:
        return {"read_ok": False}
    row = {"read_ok": True, **A1.measure(im)}
    f, seg = A3.extract(im)
    row["seg_ok"] = f is not None
    if f is not None:
        row.update(f)
        row["seg_area_frac"] = seg["area_frac"]
        row["seg_border_touch"] = seg["border_touch"]
    return row


def features_for(df, img_dir, cache: Optional[str] = None, workers: int = 4):
    """Compute (or load cached) features for every image in df."""
    if cache and os.path.exists(cache):
        cached = pd.read_csv(cache)
        if set(df.image_id) <= set(cached.image_id):
            return cached.set_index("image_id").loc[df.image_id].reset_index()
    paths = [C.find_image(img_dir, i) for i in df.image_id]
    workers = max(1, min(workers, os.cpu_count() or 1))
    if workers > 1 and len(paths) > 50:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(image_features, paths, chunksize=16)
    else:
        rows = [image_features(p) for p in paths]
    out = pd.DataFrame(rows)
    out.insert(0, "image_id", list(df.image_id))
    if cache:
        out.to_csv(cache, index=False)
    return out


def _row_to_lesion_inputs(r):
    if not r.get("read_ok", False) or not r.get("seg_ok", False) or pd.isna(r.get("asym_mean", np.nan)):
        return None, None
    f = {k: float(r[k]) for k in A3.FEATURES}
    seg = {"area_frac": float(r["seg_area_frac"]), "border_touch": float(r["seg_border_touch"])}
    return f, seg


def _quality_from_row(r, thresholds):
    if not r.get("read_ok", False):
        return {"agent": "A1_quality", "status": "FAIL", "score": 0.0,
                "reasons": ["Image could not be read"], "checks": {}, "metrics": {}}
    mt = {k: float(r[k]) for k in ("blur_laplacian_var", "field_v_mean", "glare_ratio",
                                   "obstruction_ratio", "vignette_ratio")}
    return A1.assess(metrics=mt, thresholds=thresholds)


class DermaGuard:
    """Holds the six agents configured from artifacts/calibration.json."""

    def __init__(self, calib_path=CALIB_PATH, device=None, load_cnn=True):
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"{calib_path} not found – run day2/calibrate.py first")
        with open(calib_path) as fh:
            self.cal = json.load(fh)
        self.q_thr = self.cal.get("quality_thresholds")
        self.lesion = LesionAgent(str(C.ARTIFACT_DIR / "lesion_model.json"))
        self.context = ContextAgent(str(C.ARTIFACT_DIR / "context_model.json"))
        self.coord = Coordinator(self.cal.get("fusion"), self.cal.get("safety_policy"))
        members = []
        for m in self.cal["ensemble"]:
            path = m["path"] if os.path.isabs(m["path"]) else str(C.ROOT_DIR / m["path"])
            if not os.path.exists(path) and m.get("path_abs"):
                path = m["path_abs"]
            members.append({"path": path, "weight": m.get("weight", 1.0)})
        self.diag = DiagnosisAgent(members, tta=self.cal.get("tta", 8),
                                   device=device) if load_cnn else None

    def run(self, df: pd.DataFrame, img_dir: str, workers: int = 4, feat_cache=None,
            diag_batch=None, progress=print):
        progress(f"[A2] diagnosis ensemble on {len(df)} images "
                 f"({len(self.cal['ensemble'])} model(s), TTA×{self.cal.get('tta', 8)}) …")
        if diag_batch is None:
            diag_batch = self.diag.predict(df, img_dir)
        progress("[A1/A3] quality + lesion analysis …")
        feats = features_for(df, img_dir, cache=feat_cache, workers=workers)

        recs = []
        for i, (row, fr) in enumerate(zip(df.to_dict("records"), feats.to_dict("records"))):
            q = _quality_from_row(fr, self.q_thr)
            f, seg = _row_to_lesion_inputs(fr)
            les = self.lesion.run(features=f, seg=seg, quality_score=q["score"])
            ctx = self.context.run(row.get("age"), row.get("sex"), row.get("localization"))
            d = DiagnosisAgent.describe(diag_batch, i)
            recs.append({"image_id": row["image_id"], "A1": q, "A2": d, "A3": les, "A4": ctx,
                         "true_label": row.get("dx")})

        # vectorised fusion, optional lesion-level pooling, then per-image decisions
        probs = diag_batch["probs"]
        ctx_lr = np.array([[r["A4"]["context_lr"][c] for c in C.CLASS_NAMES] for r in recs])
        ctx_rel = np.array([r["A4"]["reliability"] for r in recs])
        les_s = np.array([np.nan if r["A3"]["suspicion_score"] is None else r["A3"]["suspicion_score"]
                          for r in recs])
        les_rel = np.array([r["A3"]["reliability"] for r in recs])
        fused = fuse_arrays(probs, ctx_lr, ctx_rel, les_s, les_rel, self.coord.fusion)
        if self.cal.get("lesion_pooling") and "lesion_id" in df.columns:
            fused = pool_by_lesion(fused, df["lesion_id"].values)
        for i, r in enumerate(recs):
            r["A5_A6"] = self.coord.decide(r["A1"], r["A2"], r["A3"], r["A4"], fused_probs=fused[i])
            r["prediction"] = r["A5_A6"]["final_label"]
            r["decision"] = r["A5_A6"]["decision"]
        return recs


def pool_by_lesion(probs, lesion_ids):
    """Average log-probabilities of images that show the same lesion (no labels used)."""
    out = probs.copy()
    lp = np.log(np.clip(probs, 1e-7, 1))
    s = pd.Series(range(len(lesion_ids)), index=lesion_ids)
    for lid, idx in s.groupby(level=0):
        idx = idx.values
        if len(idx) > 1 and isinstance(lid, str) and lid:
            m = lp[idx].mean(0)
            e = np.exp(m - m.max())
            out[idx] = e / e.sum()
    return out
