"""
Agent 4 – Reasoning / Context Agent
===================================
Uses age, sex and anatomical site – only when present – to adjust how likely
each lesion class is, and explains the adjustment in words.

Model: multinomial logistic regression P(class | age, sex, site) fitted on the
TRAINING metadata only (no images, no eval labels) → artifacts/context_model.json.

Output per image
  * context_lr[c]  = P(c | metadata) / P(c)  – a likelihood ratio the Coordinator
                     can multiply into the image-based probabilities (Bayes update)
  * risk indicator – P(malignant | metadata) relative to the training base-rate
  * missing-field flags; reliability = share of fields present. With every field
    missing the agent is neutral (all ratios = 1) instead of guessing.
"""
import json
import math
import os
from typing import Optional

import numpy as np

from config import CLASS_NAMES, MALIGNANT_IDX

SITES_HEAD_NECK = {"face", "scalp", "ear", "neck"}


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in ("", "unknown", "nan", "none", "na")


def featurize(age, sex, loc, sites):
    age_missing = _is_missing(age)
    try:
        a = float(age)
    except (TypeError, ValueError):
        a, age_missing = 50.0, True
    if age_missing:
        a = 50.0
    z = (a - 50.0) / 17.0
    sx = "unknown" if _is_missing(sex) else str(sex).strip().lower()
    lc = "unknown" if _is_missing(loc) else str(loc).strip().lower()
    if lc not in sites:
        lc = "unknown"
    x = [z, z * z, float(age_missing), float(sx == "male"), float(sx == "female")]
    x += [float(lc == s) for s in sites]
    return np.array(x, float), {"age": age_missing, "sex": sx == "unknown", "localization": lc == "unknown"}


def fit(train_df, out_path):
    """Fit the metadata model on training metadata and save it as JSON."""
    from sklearn.linear_model import LogisticRegression
    sites = sorted(s for s in train_df["localization"].astype(str).str.lower().unique() if s != "unknown")
    X = np.stack([featurize(r.age, r.sex, r.localization, sites)[0] for r in train_df.itertuples()])
    y = train_df["dx"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    clf = LogisticRegression(C=0.5, max_iter=2000)
    clf.fit(X, y)
    prior = np.bincount(y, minlength=len(CLASS_NAMES)) / len(y)
    model = {"sites": sites, "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
             "prior": prior.tolist(), "n_train": int(len(y))}
    with open(out_path, "w") as fh:
        json.dump(model, fh, indent=1)
    return model


class ContextAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.m = None
        if model_path and os.path.exists(model_path):
            with open(model_path) as fh:
                self.m = json.load(fh)

    def run(self, age=None, sex=None, localization=None):
        flags = []
        present = {"age": not _is_missing(age), "sex": not _is_missing(sex),
                   "localization": not _is_missing(localization)}
        for k, ok in present.items():
            if not ok:
                flags.append(f"{k} missing")
        reliability = sum(present.values()) / 3.0

        if self.m is None or reliability == 0:
            lr = np.ones(len(CLASS_NAMES))
            post = None
        else:
            x, _ = featurize(age, sex, localization, self.m["sites"])
            logits = np.array(self.m["coef"]) @ x + np.array(self.m["intercept"])
            post = np.exp(logits - logits.max()); post /= post.sum()
            lr = post / np.array(self.m["prior"])

        prior = np.array(self.m["prior"]) if self.m else None
        if post is not None:
            p_mal = float(post[MALIGNANT_IDX].sum()); base = float(prior[MALIGNANT_IDX].sum())
            ratio = p_mal / base
        else:
            p_mal, base, ratio = None, None, 1.0
        risk = ("HIGH" if ratio >= 2.0 else "ELEVATED" if ratio >= 1.3 else
                "LOW" if ratio <= 0.6 else "AVERAGE")
        if reliability == 0:
            risk = "UNKNOWN"

        ev = []
        if present["age"]:
            a = float(age)
            if a >= 60:
                ev.append(f"Age {a:.0f}: older age raises the prior for keratoses, BCC, AK and melanoma")
            elif a < 30:
                ev.append(f"Age {a:.0f}: malignant lesions are uncommon at this age; nevi dominate")
            else:
                ev.append(f"Age {a:.0f}: middle age – near-average risk profile")
        if present["localization"]:
            loc = str(localization).lower()
            if loc in SITES_HEAD_NECK:
                ev.append(f"Site '{loc}': chronically sun-exposed head/neck → AK/BCC/lentigo more likely")
            elif loc in ("back", "trunk", "chest", "abdomen"):
                ev.append(f"Site '{loc}': typical site for nevi and also for melanoma in adults")
            elif loc in ("lower extremity",):
                ev.append(f"Site '{loc}': common site for dermatofibroma and melanoma (esp. women)")
            elif loc in ("acral", "foot", "hand"):
                ev.append(f"Site '{loc}': acral skin – patterns differ, interpret cautiously")
            else:
                ev.append(f"Site '{loc}'")
        if present["sex"]:
            ev.append(f"Sex: {str(sex).lower()}")
        if post is not None:
            top_up = [CLASS_NAMES[i] for i in np.argsort(-lr)[:2] if lr[i] > 1.15]
            if top_up:
                ev.append("Metadata favours: " + ", ".join(f"{c} (×{lr[CLASS_NAMES.index(c)]:.1f})" for c in top_up))
            ev.append(f"P(malignant | metadata) = {p_mal:.2f} vs base rate {base:.2f} (×{ratio:.2f})")
        if flags:
            ev.append("Missing: " + ", ".join(flags) + " – context weight reduced")

        return {"agent": "A4_context", "risk_indicator": risk,
                "malignant_risk_ratio": round(ratio, 3),
                "p_malignant_meta": None if p_mal is None else round(p_mal, 4),
                "context_lr": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, lr)},
                "reliability": round(reliability, 3),
                "missing_fields": flags, "evidence": ev}
