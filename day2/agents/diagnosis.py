"""
Agent 2 – Diagnosis Agent
=========================
CNN ensemble (Day-1 EfficientNet-B0 + Day-2 models) with test-time augmentation
(8 dihedral views: flips + 90° rotations).

Outputs per image: class probabilities, top-3, confidence, top1–top2 margin,
entropy, TTA stability (share of views agreeing with the final class) and
ensemble agreement (do the individual members agree?).
"""
from typing import List

import numpy as np

from config import CLASS_NAMES, CLASS_FULL_NAMES


class DiagnosisAgent:
    def __init__(self, members: List[dict], tta: int = 8, device=None):
        """members: [{"path": ckpt, "weight": w}, ...]"""
        import torch
        from common import load_checkpoint, select_device
        self.device = device or select_device()
        self.members = []
        for m in members:
            model, name, img = load_checkpoint(m["path"], self.device)
            self.members.append({"model": model, "name": name, "img": img,
                                 "weight": float(m.get("weight", 1.0)), "path": m["path"]})
        self.tta = tta

    def predict(self, df, img_dir, num_workers=2):
        from common import predict_probs
        member_probs, member_views = [], []
        for m in self.members:
            p, v = predict_probs(m["model"], df, img_dir, m["img"], self.device, tta=self.tta,
                                 num_workers=num_workers, return_views=True)
            member_probs.append(p)
            member_views.append(v)
        return self.combine(member_probs, member_views, [m["weight"] for m in self.members],
                            [f'{m["name"]}@{m["img"]}' for m in self.members])

    @staticmethod
    def combine(member_probs, member_views, weights, names):
        w = np.array(weights, float); w = w / w.sum()
        logp = sum(wi * np.log(np.clip(p, 1e-7, 1)) for wi, p in zip(w, member_probs))
        probs = np.exp(logp - logp.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
        pred = probs.argmax(1)
        tta_agree = np.mean([(v.argmax(2) == pred[None]).mean(0) for v in member_views], axis=0)
        member_pred = np.stack([p.argmax(1) for p in member_probs])         # M,N
        member_agree = (member_pred == pred[None]).mean(0)
        return {"probs": probs, "member_probs": member_probs, "member_names": names,
                "tta_agreement": tta_agree, "member_agreement": member_agree}

    @staticmethod
    def describe(batch, i):
        p = batch["probs"][i]
        order = np.argsort(-p)
        top = CLASS_NAMES[order[0]]
        ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum() / np.log(len(p)))
        members = {n: CLASS_NAMES[int(mp[i].argmax())] + f" {mp[i].max():.2f}"
                   for n, mp in zip(batch["member_names"], batch["member_probs"])}
        return {"agent": "A2_diagnosis", "predicted_class": top,
                "predicted_name": CLASS_FULL_NAMES[top],
                "confidence": round(float(p[order[0]]), 4),
                "margin": round(float(p[order[0]] - p[order[1]]), 4),
                "entropy": round(ent, 4),
                "top3": [(CLASS_NAMES[j], round(float(p[j]), 4)) for j in order[:3]],
                "probabilities": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, p)},
                "tta_agreement": round(float(batch["tta_agreement"][i]), 3),
                "ensemble_agreement": round(float(batch["member_agreement"][i]), 3),
                "members": members}
