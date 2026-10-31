"""
Output writers: submission CSV, per-image agent trace (JSONL), metrics and an
HTML report with one six-agent card per image (for the live demo).
"""
import base64
import html
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

import config as C

DECISIONS = ("AUTO_TRIAGE", "REFER", "RE_IMAGE")


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def compute_metrics(recs):
    y = [r["true_label"] for r in recs]
    p = [r["prediction"] for r in recs]
    d = [r["decision"] for r in recs]
    K = C.NUM_CLASSES
    yi = np.array([C.CLASS_TO_IDX[v] for v in y]); pi = np.array([C.CLASS_TO_IDX[v] for v in p])
    cm = np.bincount(yi * K + pi, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    per = {}
    for i, c in enumerate(C.CLASS_NAMES):
        row = cm[i].copy(); row[i] = 0
        conf = [(C.CLASS_NAMES[j], int(row[j])) for j in np.argsort(-row) if row[j] > 0][:3]
        per[c] = {"support": int(cm[i].sum()), "correct": int(cm[i, i]), "errors": int(row.sum()),
                  "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                  "f1": round(float(f1[i]), 4), "most_confused_with": conf}
    auto = [i for i, v in enumerate(d) if v == "AUTO_TRIAGE"]
    a_ok = sum(1 for i in auto if y[i] == p[i])
    return {
        "n_images": len(recs), "accuracy": round(float(tp.sum() / len(y)), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "correct_auto_triage": a_ok, "incorrect_auto_triage": len(auto) - a_ok,
        "auto_triage_precision": round(a_ok / max(1, len(auto)), 4),
        "n_refer": d.count("REFER"), "n_re_image": d.count("RE_IMAGE"),
        "confusion_matrix": {"labels": C.CLASS_NAMES, "rows_true_cols_pred": cm.tolist()},
        "per_class": per,
        "refer_by_true_class": dict(Counter(y[i] for i, v in enumerate(d) if v == "REFER")),
    }


def metrics_markdown(m, name):
    L = [f"# DermaGuard results – {name}\n",
         f"| metric | value |\n|---|---|",
         f"| images | {m['n_images']} |", f"| **Accuracy** | **{m['accuracy']:.4f}** |",
         f"| **Macro-F1** | **{m['macro_f1']:.4f}** |",
         f"| Correct AUTO_TRIAGE | {m['correct_auto_triage']} |",
         f"| Incorrect AUTO_TRIAGE | {m['incorrect_auto_triage']} |",
         f"| AUTO_TRIAGE precision | {m['auto_triage_precision']:.3f} |",
         f"| REFER | {m['n_refer']} |", f"| RE_IMAGE | {m['n_re_image']} |",
         "\n## Confusion matrix (rows = true, cols = predicted)\n",
         "| true \\ pred | " + " | ".join(C.CLASS_NAMES) + " |",
         "|---" * (C.NUM_CLASSES + 1) + "|"]
    for c, row in zip(C.CLASS_NAMES, m["confusion_matrix"]["rows_true_cols_pred"]):
        L.append(f"| **{c}** | " + " | ".join(str(v) for v in row) + " |")
    L += ["\n## Class-wise errors\n", "| class | support | correct | errors | precision | recall | F1 | most confused with |",
          "|---|---|---|---|---|---|---|---|"]
    for c, v in m["per_class"].items():
        mc = ", ".join(f"{k} ({n})" for k, n in v["most_confused_with"]) or "–"
        L.append(f"| {c} | {v['support']} | {v['correct']} | {v['errors']} | {v['precision']:.3f} | "
                 f"{v['recall']:.3f} | {v['f1']:.3f} | {mc} |")
    return "\n".join(L) + "\n"


def _thumb(path, w=180):
    im = cv2.imread(path) if path else None
    if im is None:
        return ""
    h = int(im.shape[0] * w / im.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#f6f7f9;color:#1d2330}
h1{margin:0 0 4px}.sub{color:#5b6475;margin-bottom:18px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 22px}
.kpi{background:#fff;border:1px solid #e3e6eb;border-radius:10px;padding:10px 14px;min-width:120px}
.kpi b{display:block;font-size:22px}.kpi span{color:#5b6475;font-size:12px}
table.cm{border-collapse:collapse;background:#fff;margin-bottom:22px}
table.cm td,table.cm th{border:1px solid #e3e6eb;padding:5px 9px;text-align:center;font-size:13px}
.card{background:#fff;border:1px solid #e3e6eb;border-radius:12px;padding:12px;margin:12px 0;display:flex;gap:14px}
.card img{border-radius:8px;width:180px;height:auto;align-self:flex-start}
.agents{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;flex:1}
.ag{background:#1f2533;color:#fff;border-radius:10px;padding:8px 10px;font-size:12px}
.ag .t{color:#9aa4b8;font-size:11px}.ag .v{font-size:15px;font-weight:600;margin:4px 0}
.ok{border-left:4px solid #3fb27f}.warn{border-left:4px solid #e8a33d}.bad{border-left:4px solid #e05555}
details{font-size:12px;color:#384152;margin-top:6px}.id{font-weight:600;margin-bottom:6px}
.filters button{margin-right:6px;padding:4px 10px;border-radius:14px;border:1px solid #c9ced8;background:#fff;cursor:pointer}
"""


def _card(r, images_dir):
    q, d, l, x, s = r["A1"], r["A2"], r["A3"], r["A4"], r["A5_A6"]
    saf = s["safety"]
    cls = lambda good, mid=False: "ok" if good else ("warn" if mid else "bad")
    truth = f" · true <b>{r['true_label']}</b>" if r.get("true_label") else ""
    hit = "" if not r.get("true_label") else (" ✅" if r["true_label"] == r["prediction"] else " ❌")
    ls = "n/a" if l["suspicion_score"] is None else f"{l['suspicion_score']:.2f}"
    tiles = [
        ("A1 Quality", q["status"], q["reasons"][0], cls(q["status"] == "PASS", q["status"] == "WARN")),
        ("A2 Diagnosis", f"{d['predicted_class']} · {d['confidence']:.2f}",
         "top-3: " + ", ".join(f"{c} {p:.2f}" for c, p in d["top3"]), cls(d["confidence"] >= 0.8, d["confidence"] >= 0.5)),
        ("A3 Lesion", f"{l['suspicion_level']} · {ls}", f"reliability {l['reliability_level']}; " + (l["evidence"][0] if l["evidence"] else ""),
         cls(l["suspicion_level"] == "LOW", l["suspicion_level"] in ("MODERATE", "UNKNOWN"))),
        ("A4 Context", x["risk_indicator"], (x["evidence"][0] if x["evidence"] else "no metadata"),
         cls(x["risk_indicator"] in ("LOW", "AVERAGE"), x["risk_indicator"] in ("ELEVATED", "UNKNOWN"))),
        ("A5 Safety", saf["status"].replace("_", " "), saf["summary"][:110], cls(not saf["flag"], not saf["veto"])),
        ("A6 Decision", r["decision"].replace("_", "-"), f"final label {r['prediction']}{hit}",
         cls(r["decision"] == "AUTO_TRIAGE", r["decision"] == "REFER")),
    ]
    tl = "".join(f'<div class="ag {c}"><div class="t">{html.escape(t)}</div><div class="v">{html.escape(v)}</div>'
                 f'<div>{html.escape(sub)}</div></div>' for t, v, sub, c in tiles)
    trig = "".join(f"<li><b>{t['code']}</b> → {t['action']}: {html.escape(t['justification'])}</li>" for t in saf["triggers"])
    agree = ", ".join(f"{k}: {v}" for k, v in s["agreement"].items())
    detail = ("<br>".join(html.escape(t) for t in s["rationale"]) +
              f"<br><b>Agreement:</b> {agree}" + (f"<ul>{trig}</ul>" if trig else "") +
              "<br><b>Lesion evidence:</b> " + html.escape(" | ".join(l["evidence"])) +
              "<br><b>Context evidence:</b> " + html.escape(" | ".join(x["evidence"])))
    src = _thumb(C.find_image(images_dir, r["image_id"])) if images_dir else ""
    img = f'<img src="{src}">' if src else ""
    return (f'<div class="card" data-dec="{r["decision"]}" data-hit="{int(r.get("true_label") == r["prediction"])}">{img}'
            f'<div style="flex:1"><div class="id">{r["image_id"]}{truth}</div><div class="agents">{tl}</div>'
            f'<details><summary>Traceable rationale</summary>{detail}</details></div></div>')


def write_html(recs, name, metrics, images_dir, max_cards):
    counts = Counter(r["decision"] for r in recs)
    k = [("images", len(recs))]
    if metrics:
        k += [("accuracy", f"{metrics['accuracy']:.4f}"), ("macro-F1", f"{metrics['macro_f1']:.4f}"),
              ("correct auto-triage", metrics["correct_auto_triage"]),
              ("incorrect auto-triage", metrics["incorrect_auto_triage"])]
    k += [(d, counts.get(d, 0)) for d in DECISIONS]
    kp = "".join(f'<div class="kpi"><b>{v}</b><span>{t}</span></div>' for t, v in k)
    cm_html = ""
    if metrics:
        rows = metrics["confusion_matrix"]["rows_true_cols_pred"]
        cm_html = ("<h3>Confusion matrix (rows true, columns predicted)</h3><table class='cm'><tr><th></th>" +
                   "".join(f"<th>{c}</th>" for c in C.CLASS_NAMES) + "</tr>" +
                   "".join(f"<tr><th>{c}</th>" + "".join(
                       f"<td style='background:rgba(63,178,127,{min(1, v / max(1, sum(row)) * 1.2):.2f})'>{v}</td>"
                       if i == j else f"<td style='background:rgba(224,85,85,{min(.8, v / max(1, sum(row)) * 3):.2f})'>{v}</td>"
                       for j, v in enumerate(row)) + "</tr>" for i, (c, row) in enumerate(zip(C.CLASS_NAMES, rows))) +
                   "</table>")
    sel = recs
    if len(recs) > max_cards:      # show a varied sample: every REFER/RE_IMAGE kind + some autos
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(recs))[:max_cards]
        sel = [recs[i] for i in sorted(idx)]
    cards = "".join(_card(r, images_dir) for r in sel)
    filt = ('<div class="filters">Show: <button onclick="f(\'all\')">all</button>' +
            "".join(f'<button onclick="f(\'{d}\')">{d}</button>' for d in DECISIONS) +
            ('<button onclick="f(\'wrong\')">wrong predictions</button>' if metrics else "") + "</div>")
    js = ("<script>function f(k){document.querySelectorAll('.card').forEach(c=>{c.style.display="
          "(k=='all'||c.dataset.dec==k||(k=='wrong'&&c.dataset.hit=='0'))?'flex':'none'})}</script>")
    doc = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>DermaGuard – {name}</title>"
           f"<style>{CSS}</style></head><body><h1>Agentic DermaGuard – {html.escape(name)}</h1>"
           f"<div class='sub'>Six-agent skin-lesion screening &amp; triage · cards show "
           f"{len(sel)} of {len(recs)} images</div><div class='kpis'>{kp}</div>{cm_html}{filt}{cards}{js}</body></html>")
    path = C.OUTPUT_DIR / f"{name}_report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def write_outputs(recs, name, images_dir=None, max_cards=250):
    sub = pd.DataFrame({"image_id": [r["image_id"] for r in recs],
                        "prediction": [r["prediction"] for r in recs],
                        "decision": [r["decision"] for r in recs]})
    assert sub.image_id.is_unique, "duplicate image ids!"
    assert sub.prediction.isin(C.CLASS_NAMES).all() and sub.decision.isin(DECISIONS).all()
    csv_path = C.OUTPUT_DIR / f"{name}_submission.csv"
    sub.to_csv(csv_path, index=False)
    with open(C.OUTPUT_DIR / f"{name}_agents.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=_np) + "\n")
    metrics = None
    if all(r.get("true_label") in C.CLASS_NAMES for r in recs):
        metrics = compute_metrics(recs)
        json.dump(metrics, open(C.OUTPUT_DIR / f"{name}_metrics.json", "w"), indent=1)
        (C.OUTPUT_DIR / f"{name}_metrics.md").write_text(metrics_markdown(metrics, name))
    html_path = write_html(recs, name, metrics, images_dir, max_cards)
    counts = Counter(sub.decision)
    print(f"\n[{name}] {len(sub)} images → {csv_path}")
    print("   decisions: " + ", ".join(f"{d}={counts.get(d, 0)}" for d in DECISIONS))
    if metrics:
        print(f"   accuracy={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
              f"auto-triage correct/incorrect={metrics['correct_auto_triage']}/{metrics['incorrect_auto_triage']}")
    print(f"   report: {html_path}")
    return metrics
