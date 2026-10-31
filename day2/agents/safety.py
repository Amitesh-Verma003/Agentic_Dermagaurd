"""
Agent 5 – Safety Agent
======================
Reviews the outputs of Agents 1–4 plus the Coordinator's fused prediction and
looks for reasons an automatic disposition would be unsafe.

Every trigger has a code, an action and a written justification:

  code               action     fires when
  -----------------  ---------  ----------------------------------------------------------
  QUALITY_FAIL       RE_IMAGE   A1 status FAIL (blur / exposure / glare / obstruction)
  QUALITY_UNCERTAIN  RE_IMAGE   A1 WARN *and* the model is unsure *and* the lesion could
                                not be reliably segmented → the image itself is the problem
  MALIGNANT_CLASS    REFER      final class is mel / bcc / akiec – never auto-triaged
  MALIGNANT_MASS     REFER      benign top class but P(mel+bcc+akiec) ≥ τ_mal
  LOW_CONFIDENCE     REFER      top probability < τ_conf
  LOW_MARGIN         REFER      top1 – top2 < τ_margin
  TTA_UNSTABLE       REFER      < τ_tta of the 8 augmented views agree with the final class
  MODEL_DISAGREE     REFER      ensemble members predict different classes
  FUSION_CHANGED     REFER      metadata/lesion evidence changed the CNN's top class
  LESION_CONFLICT    REFER      A3 suspicion HIGH (reliable segmentation) but benign class
  CONTEXT_CONFLICT   REFER      A4 risk HIGH and benign class with confidence < τ_ctx

Thresholds τ are tuned on the labelled eval set by calibrate.py so that
AUTO_TRIAGE keeps at least the target precision, and are stored in
artifacts/calibration.json.
"""
from config import MALIGNANT

DEFAULT_POLICY = {
    "tau_conf": 0.80, "tau_margin": 0.50, "tau_mal": 0.15, "tau_tta": 0.75,
    "tau_ctx": 0.95, "quality_uncertain_conf": 0.60,
    "use_model_disagree": True, "use_fusion_changed": True,
}


class SafetyAgent:
    def __init__(self, policy=None):
        self.p = dict(DEFAULT_POLICY)
        if policy:
            self.p.update(policy)

    def review(self, q, d, les, ctx, fused):
        """q,d,les,ctx = agent outputs; fused = coordinator fusion dict."""
        p = self.p
        trig = []

        def add(code, action, why):
            trig.append({"code": code, "action": action, "justification": why})

        label, conf, margin = fused["label"], fused["confidence"], fused["margin"]
        p_mal = fused["p_malignant"]
        benign = label not in MALIGNANT

        if q["status"] == "FAIL":
            add("QUALITY_FAIL", "RE_IMAGE", "Image quality inadequate: " + "; ".join(q["reasons"]))
        elif (q["status"] == "WARN" and conf < p["quality_uncertain_conf"]
              and les.get("reliability_level") == "LOW"):
            add("QUALITY_UNCERTAIN", "RE_IMAGE",
                f"Borderline image ({'; '.join(q['reasons'])}), low model confidence ({conf:.2f}) "
                f"and unreliable lesion segmentation – a better image is needed")

        if not benign:
            add("MALIGNANT_CLASS", "REFER",
                f"Predicted {label} (malignant/pre-malignant) – requires dermatologist review")
        elif p_mal >= p["tau_mal"]:
            add("MALIGNANT_MASS", "REFER",
                f"Benign top class but combined malignant probability {p_mal:.2f} ≥ {p['tau_mal']:.2f}")
        if conf < p["tau_conf"]:
            add("LOW_CONFIDENCE", "REFER", f"Confidence {conf:.2f} < {p['tau_conf']:.2f}")
        if margin < p["tau_margin"]:
            add("LOW_MARGIN", "REFER",
                f"Top-2 classes close ({fused['top2'][0]} vs {fused['top2'][1]}, margin {margin:.2f})")
        if d["tta_agreement"] < p["tau_tta"]:
            add("TTA_UNSTABLE", "REFER",
                f"Only {d['tta_agreement']*100:.0f}% of augmented views agree – prediction unstable")
        if p["use_model_disagree"] and d["ensemble_agreement"] < 1.0:
            add("MODEL_DISAGREE", "REFER", "Ensemble members disagree: " +
                ", ".join(f"{k}→{v}" for k, v in d["members"].items()))
        if p["use_fusion_changed"] and fused["label"] != d["predicted_class"]:
            add("FUSION_CHANGED", "REFER",
                f"Image-only model said {d['predicted_class']}, combined evidence says {label}")
        if benign and les.get("suspicion_level") == "HIGH" and les.get("reliability_level") in ("HIGH", "MEDIUM"):
            add("LESION_CONFLICT", "REFER",
                f"Model predicts benign {label} but ABC analysis is highly suspicious "
                f"(score {les['suspicion_score']:.2f})")
        if benign and ctx.get("risk_indicator") == "HIGH" and conf < p["tau_ctx"]:
            add("CONTEXT_CONFLICT", "REFER",
                f"High-risk context (×{ctx['malignant_risk_ratio']:.1f} malignant risk) with benign "
                f"prediction at confidence {conf:.2f}")

        veto = any(t["action"] == "RE_IMAGE" for t in trig)
        flag = bool(trig)
        if veto:
            status = "VETO_RE_IMAGE"
        elif flag:
            status = "SAFETY_FLAG"
        else:
            status = "CLEAR"
        summary = ("No conflicts – automatic disposition is safe" if not trig else
                   "; ".join(t["code"] for t in trig))
        return {"agent": "A5_safety", "status": status, "flag": flag, "veto": veto,
                "triggers": trig, "summary": summary}
