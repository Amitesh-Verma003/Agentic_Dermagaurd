"""
Agent 6 – Coordinator Agent
===========================
Receives every agent's output and applies one documented policy.

Step 1 – evidence fusion (final label)
    log P_final(c) ∝ log P_cnn(c)                                  (A2, ensemble + TTA)
                   + w_ctx · rel_ctx · log LR_ctx(c)               (A4, metadata Bayes update)
                   + w_les · rel_les · logit(suspicion) · 1[c malignant]   (A3, ABC evidence)
                   + b_c                                           (per-class bias, tuned on eval
                                                                     to balance Accuracy & Macro-F1)
    w_ctx, w_les and b_c are fitted on the labelled eval set by calibrate.py.
    rel_* are the agents' own reliabilities, so missing metadata or a failed
    segmentation automatically counts for less.

Step 2 – Safety review (A5) on the fused prediction.

Step 3 – triage decision, in strict priority order
    1. any RE_IMAGE trigger from A5 (image unusable)        → RE_IMAGE
    2. any REFER trigger from A5 (suspicion / uncertainty /
       conflicting evidence)                                → REFER
    3. otherwise (reliable, consistent, benign evidence)    → AUTO_TRIAGE
    Every image keeps a diagnosis prediction whatever the decision.

Step 4 – traceable rationale: one line per agent, an agreement table
    (SUPPORTS / CONCERN / NEUTRAL) and the rule that decided.
"""
import math

import numpy as np

from config import CLASS_NAMES, MALIGNANT, MALIGNANT_IDX
from agents.safety import SafetyAgent

DEFAULT_FUSION = {"w_ctx": 0.0, "w_les": 0.0, "bias": [0.0] * len(CLASS_NAMES)}


def fuse_arrays(probs, ctx_lr, ctx_rel, les_score, les_rel, fusion):
    """Vectorised fusion used by both the pipeline and calibration.
    probs (N,C); ctx_lr (N,C); ctx_rel (N,); les_score (N,) in (0,1) or nan; les_rel (N,)."""
    logp = np.log(np.clip(probs, 1e-7, 1.0))
    logp = logp + fusion["w_ctx"] * ctx_rel[:, None] * np.log(np.clip(ctx_lr, 1e-4, 1e4))
    s = np.nan_to_num(les_score, nan=0.5)
    lo = np.log(np.clip(s, 1e-3, 1 - 1e-3) / np.clip(1 - s, 1e-3, 1))
    add = np.zeros_like(logp)
    add[:, MALIGNANT_IDX] = (fusion["w_les"] * les_rel * lo)[:, None]
    logp = logp + add + np.array(fusion["bias"])[None]
    out = np.exp(logp - logp.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


class Coordinator:
    def __init__(self, fusion=None, policy=None):
        self.fusion = dict(DEFAULT_FUSION)
        if fusion:
            self.fusion.update(fusion)
        self.safety = SafetyAgent(policy)

    def fuse_one(self, d, les, ctx, fused_probs=None):
        if fused_probs is None:
            probs = np.array([[d["probabilities"][c] for c in CLASS_NAMES]])
            lr = np.array([[ctx["context_lr"][c] for c in CLASS_NAMES]])
            s = les["suspicion_score"] if les["suspicion_score"] is not None else float("nan")
            fused_probs = fuse_arrays(probs, lr, np.array([ctx["reliability"]]),
                                      np.array([s]), np.array([les["reliability"]]), self.fusion)[0]
        p = fused_probs
        order = np.argsort(-p)
        return {"label": CLASS_NAMES[order[0]], "confidence": float(p[order[0]]),
                "margin": float(p[order[0]] - p[order[1]]),
                "top2": (CLASS_NAMES[order[0]], CLASS_NAMES[order[1]]),
                "p_malignant": float(p[MALIGNANT_IDX].sum()),
                "probabilities": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)}}

    def decide(self, q, d, les, ctx, fused_probs=None):
        fused = self.fuse_one(d, les, ctx, fused_probs)
        saf = self.safety.review(q, d, les, ctx, fused)
        if saf["veto"]:
            decision, rule = "RE_IMAGE", "Rule 1: Safety veto – image quality inadequate"
        elif saf["flag"]:
            decision, rule = "REFER", "Rule 2: Safety flag – " + saf["summary"]
        else:
            decision, rule = "AUTO_TRIAGE", "Rule 3: reliable, consistent, benign evidence"

        label = fused["label"]
        benign = label not in MALIGNANT
        agree = {
            "A1_quality": "SUPPORTS" if q["status"] == "PASS" else "CONCERN",
            "A2_diagnosis": "SUPPORTS" if d["predicted_class"] == label else "CONCERN",
            "A3_lesion": ("NEUTRAL" if les.get("suspicion_level") in ("UNKNOWN", None)
                          or les.get("reliability_level") == "LOW" else
                          "SUPPORTS" if (les["suspicion_level"] == "HIGH") == (not benign)
                          or les["suspicion_level"] == "MODERATE" else "CONCERN"),
            "A4_context": ("NEUTRAL" if ctx["risk_indicator"] in ("UNKNOWN", "AVERAGE") else
                           "SUPPORTS" if (ctx["risk_indicator"] in ("HIGH", "ELEVATED")) == (not benign)
                           else "CONCERN"),
            "A5_safety": "SUPPORTS" if not saf["flag"] else "CONCERN",
        }
        s_txt = "n/a" if les.get("suspicion_score") is None else "%.2f" % les["suspicion_score"]
        trace = [
            f"A1 Quality: {q['status']} (score {q['score']:.2f}) – {q['reasons'][0]}",
            f"A2 Diagnosis: {d['predicted_class']} {d['confidence']:.2f} "
            f"(margin {d['margin']:.2f}, TTA {d['tta_agreement']*100:.0f}% stable)",
            f"A3 Lesion: suspicion {les.get('suspicion_level')} ({s_txt}), "
            f"reliability {les.get('reliability_level')}",
            f"A4 Context: risk {ctx['risk_indicator']} (×{ctx['malignant_risk_ratio']:.2f})"
            + (f", missing: {', '.join(ctx['missing_fields'])}" if ctx["missing_fields"] else ""),
            f"Fusion: {label} {fused['confidence']:.2f}, P(malignant) {fused['p_malignant']:.2f}",
            f"A5 Safety: {saf['status']} – {saf['summary']}",
            f"A6 Decision: {decision} ({rule})",
        ]
        return {"agent": "A6_coordinator", "final_label": label, "decision": decision,
                "rule": rule, "fused": fused, "agreement": agree, "safety": saf,
                "rationale": trace}
