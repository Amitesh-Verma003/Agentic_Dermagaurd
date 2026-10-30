"""
inspect_data.py – Quick sanity check: class counts for train & val.
Run this first to verify everything loads correctly.
"""
from config import TRAIN_CSV, TRAIN_DIR, VAL_CSV, VAL_DIR, CLASSES
from dataset import SkinLesionDataset, train_transforms, val_transforms


def main():
    print("=" * 55)
    print("  HAM10000 Data Inspection")
    print("=" * 55)

    print("\n> Loading TRAIN set ...")
    train_ds = SkinLesionDataset(TRAIN_CSV, TRAIN_DIR, train_transforms)
    train_counts = train_ds.class_counts()

    print("\n> Loading VAL set ...")
    val_ds = SkinLesionDataset(VAL_CSV, VAL_DIR, val_transforms)
    val_counts = val_ds.class_counts()

    # ── Pretty table ──────────────────────────────────────────────────
    print(f"\n{'Class':<8} {'Train':>7} {'Val':>7} {'Train %':>8}")
    print("-" * 34)
    total_train = sum(train_counts.values())
    for c in CLASSES:
        pct = 100.0 * train_counts[c] / total_train if total_train else 0
        print(f"{c:<8} {train_counts[c]:>7,} {val_counts[c]:>7,} {pct:>7.1f}%")
    total_val = sum(val_counts.values())
    print("-" * 34)
    print(f"{'TOTAL':<8} {total_train:>7,} {total_val:>7,}")

    # ── Quick sample check ────────────────────────────────────────────
    img, lbl, iid = train_ds[0]
    print(f"\nSample check  ->  id={iid}, label_idx={lbl}, "
          f"tensor shape={tuple(img.shape)}")
    print("\n[OK] Data inspection complete.")


if __name__ == "__main__":
    main()
