"""
Live panel demo – runs all six agents on a few images and prints the
agent-by-agent cards plus the traceable rationale.

  python day2/demo.py --data "$DG_EVAL_DIR" --ids ISIC_0024306 ISIC_0024307
  python day2/demo.py --data "$DG_EVAL_DIR" --random 4
  python day2/demo.py --image /path/to/any_lesion.jpg --age 63 --sex male --site back
"""
import argparse
import multiprocessing as mp
import os
import textwrap

import pandas as pd

import config as C
from common import image_dir_of, load_table
from pipeline import DermaGuard

W = 26


def box(title, value, note):
    lines = [title, value] + textwrap.wrap(note, W - 2)[:3]
    return [l[:W - 2].ljust(W - 2) for l in lines] + [" " * (W - 2)] * (5 - len(lines))


def show(r):
    q, d, l, x, s = r["A1"], r["A2"], r["A3"], r["A4"], r["A5_A6"]
    ls = "n/a" if l["suspicion_score"] is None else f"{l['suspicion_score']:.2f}"
    cols = [
        box("A1 QUALITY", q["status"], q["reasons"][0]),
        box("A2 DIAGNOSIS", f"{d['predicted_class']} • {d['confidence']:.2f}",
            "top3 " + " ".join(f"{c}:{p:.2f}" for c, p in d["top3"])),
        box("A3 LESION", f"{l['suspicion_level']} • {ls}", f"reliab. {l['reliability_level']}; " + l["evidence"][0]),
        box("A4 CONTEXT", x["risk_indicator"], x["evidence"][0] if x["evidence"] else "no metadata"),
        box("A5 SAFETY", s["safety"]["status"], s["safety"]["summary"]),
        box("A6 DECISION", r["decision"], f"final label {r['prediction']}"),
    ]
    truth = f"   (true label: {r['true_label']})" if r.get("true_label") else ""
    print("\n" + "═" * (W * 6 + 1))
    print(f" {r['image_id']}{truth}")
    print("┌" + "┬".join("─" * W for _ in cols) + "┐")
    for i in range(5):
        print("│" + "│".join(" " + c[i] + " " for c in cols) + "│")
    print("└" + "┴".join("─" * W for _ in cols) + "┘")
    print(" Rationale:")
    for t in s["rationale"]:
        print("   • " + t)
    print("   • Agreement: " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in s["agreement"].items()))
    for t in s["safety"]["triggers"]:
        print(f"   ⚑ {t['code']} → {t['action']}: {t['justification']}")
    print("   Lesion evidence: " + " | ".join(l["evidence"]))
    print("   Context evidence: " + " | ".join(x["evidence"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.EVAL_DIR)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--image", help="single image file")
    ap.add_argument("--age"); ap.add_argument("--sex"); ap.add_argument("--site")
    a = ap.parse_args()

    if a.image:
        img_dir = os.path.dirname(os.path.abspath(a.image))
        df = pd.DataFrame([{"image_id": os.path.splitext(os.path.basename(a.image))[0],
                            "age": float(a.age) if a.age else None, "sex": a.sex, "localization": a.site}])
    else:
        df = load_table(a.data)
        img_dir = image_dir_of(a.data)
        if a.ids:
            df = df[df.image_id.isin(a.ids)]
        elif a.random:
            df = df.sample(a.random)
        else:
            df = df.sample(3, random_state=7)
    dg = DermaGuard()
    recs = dg.run(df.reset_index(drop=True), img_dir, workers=1)
    for r in recs:
        show(r)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
