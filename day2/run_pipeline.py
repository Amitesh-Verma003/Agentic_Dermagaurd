"""
Run the full six-agent DermaGuard system on any image set.

  # hidden test (no labels) → submission CSV + per-image traces + HTML report
  python day2/run_pipeline.py --data "/path/to/hidden" --name hidden

  # labelled eval pack → also accuracy, macro-F1, confusion matrix, class errors
  python day2/run_pipeline.py --data "$DG_EVAL_DIR" --name eval

--data may be a folder with images/ + metadata.csv, or a plain folder of images
(metadata then treated as missing – the Context Agent becomes neutral).
"""
import argparse
import multiprocessing as mp
import os
import time

import config as C
from common import image_dir_of, load_table
from pipeline import DermaGuard
from reporting import write_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--metadata", default=None, help="CSV with image_id[,age,sex,localization,lesion_id]")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cards", type=int, default=250)
    a = ap.parse_args()

    t0 = time.time()
    df = load_table(a.data, a.metadata)
    img_dir = image_dir_of(a.data)
    missing = [i for i in df.image_id if C.find_image(img_dir, i) is None]
    if missing:
        raise SystemExit(f"{len(missing)} listed images not found in {img_dir}, e.g. {missing[:5]}")
    on_disk = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    if on_disk != len(df):
        print(f"⚠️  {on_disk} image files on disk but {len(df)} rows in metadata – using the metadata IDs")
    name = a.name or os.path.basename(os.path.normpath(a.data)).replace(" ", "_")
    print(f"[run] {len(df)} images from {img_dir}  (labels: {'yes' if 'dx' in df else 'no'})")

    dg = DermaGuard()
    recs = dg.run(df, img_dir, workers=a.workers)
    write_outputs(recs, name, images_dir=img_dir, max_cards=a.max_cards)
    print(f"[run] finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
