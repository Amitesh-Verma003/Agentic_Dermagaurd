"""
Agentic DermaGuard – calibration on the labelled eval set
=========================================================
Run after training. It:
  1. Measures quality metrics + ABC features on the TRAINING images
       → data-justified quality thresholds (A1)
       → logistic suspicion model on ABC features (A3)        artifacts/lesion_model.json
  2. Fits the metadata model on TRAINING metadata (A4)         artifacts/context_model.json
  3. Runs every CNN checkpoint on the eval set with 8× TTA (A2) and picks the
     best ensemble.
  4. Fits the Coordinator's evidence fusion (A6): context weight, lesion weight
     and per-class biases, maximising ½·Accuracy + ½·Macro-F1. Reports an honest
     5-fold cross-validated estimate as well.
  5. Tunes the Safety Agent thresholds (A5) so AUTO_TRIAGE keeps ≥ target
     precision while auto-triaging as many images as possible.
  6. Writes artifacts/calibration.json, an ablation report showing how each
     agent changes the result, and the full eval-set outputs.

Usage:  python day2/calibrate.py [--target-precision 0.96] [--ckpts a.pth b.pth]
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config as C
from agents import context as A4mod
from agents import quality as A1
from agents.context import ContextAgent
from agents.coordinator import fuse_arrays
from agents.diagnosis import DiagnosisAgent
from agents.lesion import FEATURES, LesionAgent
from agents.safety import DEFAULT_POLICY
from common import image_dir_of, load_checkpoint, load_table, predict_probs, select_device
from pipeline import CALIB_PATH, DermaGuard, features_for, _quality_from_row, _row_to_lesion_inputs

NV = C.CLASS_TO_IDX["nv"]
K = C.NUM_CLASSES


# ── metrics ──────────────────────────────────────────────────────────────
def conf_mat(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def acc_f1(y, p):
    cm = conf_mat(y, p)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    return tp.sum() / len(y), f1.mean()


def objective(y, p):
    a, f = acc_f1(y, p)
    return 0.5 * a + 0.5 * f


def tune_bias(logp, y, passes=3, grid=np.round(np.arange(-2.0, 2.01, 0.1), 2)):
    b = np.zeros(K)
    best = objective(y, (logp + b).argmax(1))
    for _ in range(passes):
        improved = False
        for c in range(K):
            if c == NV:
                continue                       # nv is the reference class
            for v in grid:
                bb = b.copy(); bb[c] = v
                s = objective(y, (logp + bb).argmax(1))
                if s > best + 1e-9:
                    best, b, improved = s, bb, True
        if not improved:
            break
    return b, best


# ── vectorised mirror of agents/safety.py (verified against it at the end) ──
def decisions_vec(S, pol):
    benign = ~np.isin(S["label"], C.MALIGNANT_IDX)
    re_img = (S["q"] == "FAIL") | ((S["q"] == "WARN") & (S["conf"] < pol["quality_uncertain_conf"])
                                   & (S["les_rel"] == "LOW"))
    refer = (~benign) | (benign & (S["p_mal"] >= pol["tau_mal"])) | (S["conf"] < pol["tau_conf"]) \
        | (S["margin"] < pol["tau_margin"]) | (S["tta"] < pol["tau_tta"]) \
        | (benign & (S["les_lvl"] == "HIGH") & np.isin(S["les_rel"], ["HIGH", "MEDIUM"])) \
        | (benign & (S["ctx"] == "HIGH") & (S["conf"] < pol["tau_ctx"]))
    if pol["use_model_disagree"]:
        refer |= S["member_agree"] < 1.0
    if pol["use_fusion_changed"]:
        refer |= S["label"] != S["cnn_label"]
    dec = np.where(re_img, "RE_IMAGE", np.where(refer, "REFER", "AUTO_TRIAGE"))
    return dec


def auto_stats(dec, label, y):
    a = dec == "AUTO_TRIAGE"
    correct = int((a & (label == y)).sum())
    return correct, int(a.sum()) - correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=C.TRAIN_DIR)
    ap.add_argument("--eval-dir", default=C.EVAL_DIR)
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="checkpoints to consider (default: Day-1 + day2/artifacts/*.pth)")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--target-precision", type=float, default=0.96)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time()
    rep = []                                              # markdown report lines
    log = lambda s="": (print(s, flush=True), rep.append(s))

    tr = load_table(a.train_dir)
    ev = load_table(a.eval_dir)
    y = ev["dx"].map(C.CLASS_TO_IDX).values
    log(f"# DermaGuard calibration report\n\ntrain = {len(tr)} images, eval = {len(ev)} images\n")

    # ── 1. training-image features → A1 thresholds + A3 suspicion model ──────
    print("[1/6] quality + ABC features on training images (cached after first run) …")
    ftr = features_for(tr, image_dir_of(a.train_dir), cache=str(C.ARTIFACT_DIR / "features_train.csv"),
                       workers=a.workers)
    qthr = A1.thresholds_from_metrics(ftr[ftr.read_ok])
    log("## Agent 1 – quality thresholds (from training-image percentiles; FAIL≈worst 1 %, WARN≈worst 5 %)\n")
    for k, v in qthr.items():
        log(f"- {k}: warn {v[0]:.3f}, fail {v[1]:.3f}")

    ok = ftr.seg_ok.fillna(False).astype(bool).values
    Xl = ftr.loc[ok, FEATURES].values.astype(float)
    yl = np.isin(tr.dx.values[ok], list(C.MALIGNANT)).astype(int)
    mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
    groups = tr.lesion_id.values[ok] if "lesion_id" in tr else np.arange(ok.sum())
    oof = np.zeros(len(yl))
    for trn, tst in GroupKFold(5).split(Xl, yl, groups):
        m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
        m.fit((Xl[trn] - mu) / sd, yl[trn]); oof[tst] = m.predict_proba((Xl[tst] - mu) / sd)[:, 1]
    lm = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000).fit((Xl - mu) / sd, yl)
    hi = float(np.percentile(oof[yl == 0], 85)); lo = float(np.percentile(oof[yl == 0], 50))
    lesion_model = {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
                    "coef": lm.coef_[0].tolist(), "intercept": float(lm.intercept_[0]),
                    "levels": [round(lo, 3), round(hi, 3)], "cv_auc": float(roc_auc_score(yl, oof)),
                    "segmented": int(ok.sum()), "total": int(len(ok))}
    json.dump(lesion_model, open(C.ARTIFACT_DIR / "lesion_model.json", "w"), indent=1)
    log(f"\n## Agent 3 – lesion suspicion model\n\nsegmented {ok.sum()}/{len(ok)} training images; "
        f"malignant-vs-benign CV AUC = {lesion_model['cv_auc']:.3f}; levels LOW<{lo:.2f}≤MODERATE<{hi:.2f}≤HIGH")
    top = sorted(zip(FEATURES, lm.coef_[0]), key=lambda t: -abs(t[1]))[:6]
    log("strongest cues: " + ", ".join(f"{f} ({w:+.2f})" for f, w in top))

    # ── 2. context model ─────────────────────────────────────────────────────
    print("[2/6] metadata model …")
    A4mod.fit(tr, C.ARTIFACT_DIR / "context_model.json")
    ctx_agent = ContextAgent(str(C.ARTIFACT_DIR / "context_model.json"))
    les_agent = LesionAgent(str(C.ARTIFACT_DIR / "lesion_model.json"))

    # ── eval features + agent outputs ───────────────────────────────────────
    print("[3/6] eval-set features …")
    fev = features_for(ev, image_dir_of(a.eval_dir), cache=str(C.ARTIFACT_DIR / "features_eval.csv"),
                       workers=a.workers)
    Q, L3, X4 = [], [], []
    for row, fr in zip(ev.to_dict("records"), fev.to_dict("records")):
        q = _quality_from_row(fr, qthr)
        f, seg = _row_to_lesion_inputs(fr)
        Q.append(q)
        L3.append(les_agent.run(features=f, seg=seg, quality_score=q["score"]))
        X4.append(ctx_agent.run(row.get("age"), row.get("sex"), row.get("localization")))
    ctx_lr = np.array([[c["context_lr"][k] for k in C.CLASS_NAMES] for c in X4])
    ctx_rel = np.array([c["reliability"] for c in X4])
    les_s = np.array([np.nan if l["suspicion_score"] is None else l["suspicion_score"] for l in L3])
    les_rel = np.array([l["reliability"] for l in L3])
    mal_true = np.isin(y, C.MALIGNANT_IDX)
    m_ok = ~np.isnan(les_s)
    log(f"\nA3 on eval: segmented {m_ok.sum()}/{len(ev)}, AUC malignant = "
        f"{roc_auc_score(mal_true[m_ok], les_s[m_ok]):.3f}")
    p_meta = np.array([c["p_malignant_meta"] or 0 for c in X4])
    log(f"A4 on eval: metadata-only AUC malignant = {roc_auc_score(mal_true, p_meta):.3f}")

    # ── 3. CNN members ───────────────────────────────────────────────────────
    print("[4/6] CNN checkpoints on eval (8× TTA, cached) …")
    ckpts = a.ckpts or ([str(C.DAY1_CKPT)] if C.DAY1_CKPT.exists() else []) + \
        sorted(glob.glob(str(C.ARTIFACT_DIR / "*.pth")))
    device = select_device()
    cache_dir = C.ARTIFACT_DIR / "cache"; cache_dir.mkdir(exist_ok=True)
    members = []
    log("\n## Agent 2 – diagnosis models on eval\n\n| model | no TTA acc | no TTA F1 | TTA acc | TTA F1 |\n|---|---|---|---|---|")
    for ck in ckpts:
        key = cache_dir / f"eval_{os.path.basename(ck)}_{int(os.path.getmtime(ck))}_{a.tta}.npz"
        if key.exists():
            z = np.load(key); probs, views = z["probs"], z["views"]
            name = str(z["name"])
        else:
            model, mname, img = load_checkpoint(ck, device)
            probs, views = predict_probs(model, ev, image_dir_of(a.eval_dir), img, device,
                                         tta=a.tta, return_views=True)
            name = f"{mname}@{img}"
            np.savez_compressed(key, probs=probs, views=views.astype(np.float16), name=name)
            del model
        a0, f0 = acc_f1(y, views[0].argmax(1)); a1, f1 = acc_f1(y, probs.argmax(1))
        log(f"| {os.path.basename(ck)} ({name}) | {a0:.4f} | {f0:.4f} | {a1:.4f} | {f1:.4f} |")
        members.append({"path": ck, "name": name, "probs": probs, "views": views.astype(np.float32)})
    if not members:
        raise SystemExit("No checkpoints found. Train first (day2/train_v2.py).")

    best = None
    for r in range(1, min(len(members), 5) + 1):
        for sub in itertools.combinations(range(len(members)), r):
            b = DiagnosisAgent.combine([members[i]["probs"] for i in sub], [members[i]["views"] for i in sub],
                                       [1.0] * len(sub), [members[i]["name"] for i in sub])
            s = objective(y, b["probs"].argmax(1))
            if best is None or s > best[0] + 1e-4:
                best = (s, sub, b)
    _, sub, batch = best
    ens_names = [members[i]["name"] for i in sub]
    ea, ef = acc_f1(y, batch["probs"].argmax(1))
    log(f"\nSelected ensemble: {ens_names} → acc {ea:.4f}, macro-F1 {ef:.4f}")

    # ── 4. Coordinator fusion ────────────────────────────────────────────────
    print("[5/6] fitting coordinator fusion …")
    zero = {"w_ctx": 0.0, "w_les": 0.0, "bias": [0.0] * K}
    grid = []
    for wc in (0.0, 0.25, 0.5, 0.75, 1.0):
        for wl in (0.0, 0.1, 0.2, 0.35, 0.5):
            fz = fuse_arrays(batch["probs"], ctx_lr, ctx_rel, les_s, les_rel, {**zero, "w_ctx": wc, "w_les": wl})
            grid.append((objective(y, fz.argmax(1)), wc, wl, np.log(fz)))
    # choose weights on the bias-tuned objective (prefer smaller weights on ties)
    scored = []
    for s0, wc, wl, lp in grid:
        b, s = tune_bias(lp, y, passes=2)
        scored.append((s - 0.0005 * (wc + wl), wc, wl))
    _, wc, wl = max(scored)
    lp = np.log(fuse_arrays(batch["probs"], ctx_lr, ctx_rel, les_s, les_rel, {**zero, "w_ctx": wc, "w_les": wl}))
    bias, _ = tune_bias(lp, y)
    fusion = {"w_ctx": wc, "w_les": wl, "bias": [round(float(v), 3) for v in bias]}

    def fused_with(fz_cfg, pool=False):
        fz = fuse_arrays(batch["probs"], ctx_lr, ctx_rel, les_s, les_rel, fz_cfg)
        if pool:
            from pipeline import pool_by_lesion
            fz = pool_by_lesion(fz, ev["lesion_id"].values)
        return fz

    abl = [("A2 ensemble (CNN + TTA)", fused_with(zero)),
           ("+ A4 context fusion", fused_with({**zero, "w_ctx": wc})),
           ("+ A3 lesion fusion", fused_with({**zero, "w_ctx": wc, "w_les": wl})),
           ("+ per-class bias (final fusion)", fused_with(fusion))]
    pooling = False
    if "lesion_id" in ev.columns and ev.lesion_id.duplicated().any():
        fp = fused_with(fusion, pool=True)
        if objective(y, fp.argmax(1)) > objective(y, abl[-1][1].argmax(1)) + 0.002:
            pooling = True
            abl.append(("+ pool images of the same lesion", fp))

    # honest estimate: 5-fold grouped CV of the bias tuning
    groups = ev.lesion_id.values if "lesion_id" in ev else np.arange(len(ev))
    oof_pred = np.zeros(len(y), int)
    for trn, tst in GroupKFold(5).split(lp, y, groups):
        b, _ = tune_bias(lp[trn], y[trn])
        oof_pred[tst] = (lp[tst] + b).argmax(1)
    cva, cvf = acc_f1(y, oof_pred)

    log("\n## Agent 6 – how each agent changes the diagnosis (eval set)\n\n| stage | accuracy | macro-F1 |\n|---|---|---|")
    log(f"| Day-1 baseline (reported) | 0.7889 | – |")
    for name, fz in abl:
        aa, ff = acc_f1(y, fz.argmax(1))
        log(f"| {name} | {aa:.4f} | {ff:.4f} |")
    log(f"| final, 5-fold cross-validated (honest estimate) | {cva:.4f} | {cvf:.4f} |")
    log(f"\nfusion: w_ctx={wc}, w_les={wl}, bias={dict(zip(C.CLASS_NAMES, fusion['bias']))}, "
        f"lesion pooling={'on' if pooling else 'off'}")

    final = abl[-1][1]
    label = final.argmax(1)

    # ── 5. Safety thresholds ────────────────────────────────────────────────
    print("[6/6] tuning safety thresholds …")
    srt = np.sort(final, 1)
    S = {"label": label, "cnn_label": batch["probs"].argmax(1), "conf": srt[:, -1],
         "margin": srt[:, -1] - srt[:, -2], "p_mal": final[:, C.MALIGNANT_IDX].sum(1),
         "tta": batch["tta_agreement"], "member_agree": batch["member_agreement"],
         "q": np.array([q["status"] for q in Q]), "les_lvl": np.array([l["suspicion_level"] for l in L3]),
         "les_rel": np.array([l["reliability_level"] for l in L3]),
         "ctx": np.array([c["risk_indicator"] for c in X4])}
    cands = []
    for tc in np.arange(0.40, 0.96, 0.05):
        for tm in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            for tg in (0.0, 0.2, 0.3, 0.4, 0.5):
                for tt in (0.5, 0.625, 0.75, 0.875):
                    for md in (True, False):
                        pol = {**DEFAULT_POLICY, "tau_conf": round(float(tc), 2), "tau_mal": tm,
                               "tau_margin": tg, "tau_tta": tt, "use_model_disagree": md}
                        dec = decisions_vec(S, pol)
                        c_ok, c_bad = auto_stats(dec, label, y)
                        prec = c_ok / max(1, c_ok + c_bad)
                        cands.append((prec >= a.target_precision, c_ok, -c_bad, pol))
    feas = [c for c in cands if c[0]]
    if not feas:
        print(f"   no setting reaches precision {a.target_precision}; using the most precise one")
        feas = sorted(cands, key=lambda c: c[1] / max(1, c[1] - c[2]), reverse=True)[:1]
    # most correct auto-triages; ties → fewer errors, then the more conservative thresholds
    _, _, _, policy = max(feas, key=lambda c: (c[1], c[2], c[3]["tau_conf"] + c[3]["tau_margin"]
                                               + c[3]["tau_tta"] - c[3]["tau_mal"]))
    dec = decisions_vec(S, policy)
    c_ok, c_bad = auto_stats(dec, label, y)
    log(f"\n## Agent 5 – safety policy (target AUTO_TRIAGE precision ≥ {a.target_precision})\n")
    log(", ".join(f"{k}={v}" for k, v in policy.items()))
    log(f"\nAUTO_TRIAGE {c_ok + c_bad} (correct {c_ok}, incorrect {c_bad}, precision "
        f"{c_ok / max(1, c_ok + c_bad):.3f}) · REFER {(dec == 'REFER').sum()} · RE_IMAGE {(dec == 'RE_IMAGE').sum()}")

    # leave-one-trigger-out: what each safety rule contributes
    log("\n| safety rule switched off | extra auto-triaged | of which wrong |\n|---|---|---|")
    base_auto = dec == "AUTO_TRIAGE"
    for nm, change in (("LOW_CONFIDENCE", {"tau_conf": 0.0}), ("MALIGNANT_MASS", {"tau_mal": 1.01}),
                       ("LOW_MARGIN", {"tau_margin": 0.0}), ("TTA_UNSTABLE", {"tau_tta": 0.0}),
                       ("MODEL_DISAGREE", {"use_model_disagree": False}),
                       ("FUSION_CHANGED", {"use_fusion_changed": False}),
                       ("CONTEXT_CONFLICT", {"tau_ctx": 0.0})):
        d2 = decisions_vec(S, {**policy, **change}) == "AUTO_TRIAGE"
        extra = d2 & ~base_auto
        log(f"| {nm} | {extra.sum()} | {(extra & (label != y)).sum()} |")
    S2 = dict(S); S2["les_lvl"] = np.full(len(y), "LOW")
    extra = (decisions_vec(S2, policy) == "AUTO_TRIAGE") & ~base_auto
    log(f"| LESION_CONFLICT | {extra.sum()} | {(extra & (label != y)).sum()} |")

    # ── save calibration ────────────────────────────────────────────────────
    rel = lambda p: os.path.relpath(p, C.ROOT_DIR) if str(p).startswith(str(C.ROOT_DIR)) else str(p)
    cal = {"ensemble": [{"path": rel(members[i]["path"]), "weight": 1.0, "name": members[i]["name"]} for i in sub],
           "tta": a.tta, "fusion": fusion, "lesion_pooling": pooling, "safety_policy": policy,
           "quality_thresholds": qthr, "target_precision": a.target_precision,
           "eval_metrics": {"accuracy": float(acc_f1(y, label)[0]), "macro_f1": float(acc_f1(y, label)[1]),
                            "cv_accuracy": float(cva), "cv_macro_f1": float(cvf)}}
    for m in cal["ensemble"]:
        if not os.path.isabs(m["path"]):
            m["path_abs"] = str(C.ROOT_DIR / m["path"])
    json.dump(cal, open(CALIB_PATH, "w"), indent=1)
    log(f"\nSaved {CALIB_PATH}")

    # ── verify with the real agents + write the full eval outputs ───────────
    from reporting import write_outputs
    dg = DermaGuard(load_cnn=False)
    recs = dg.run(ev, image_dir_of(a.eval_dir), feat_cache=str(C.ARTIFACT_DIR / "features_eval.csv"),
                  diag_batch=batch, progress=lambda s: None)
    agent_dec = np.array([r["decision"] for r in recs])
    mismatch = int((agent_dec != dec).sum())
    if mismatch:
        print(f"   ⚠️  {mismatch} decisions differ between tuning mirror and agents (check safety.py edits)")
    open(C.OUTPUT_DIR / "calibration_report.md", "w").write("\n".join(rep) + "\n")
    write_outputs(recs, "eval", images_dir=image_dir_of(a.eval_dir), max_cards=120)
    print(f"\nDone in {time.time() - t_start:.0f}s. Report → {C.OUTPUT_DIR / 'calibration_report.md'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
