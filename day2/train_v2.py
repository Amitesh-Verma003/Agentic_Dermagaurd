"""
Agentic DermaGuard – Diagnosis Agent v2 training
================================================
What changed vs Day 1 (78.9% on the eval pack):
  1. Trains on ALL 7,762 training images (Day 1 held 20% back with a random,
     non-lesion-disjoint split → the 84% internal score was optimistic).
  2. Validates on the official lesion-disjoint eval pack → honest model selection.
  3. Softer class weighting (n_c^-power, default 0.25 instead of full inverse
     frequency). Full inverse weights pushed nv recall down to 87%, and nv is
     67% of the data, so accuracy suffered. Class balance is restored later by
     per-class bias tuning in calibrate.py, which keeps Macro-F1 high.
  4. Label smoothing, stronger dermoscopy augmentation (full rotations, crops,
     random erasing), warm-up + cosine LR per step, EMA weights.
  5. Bigger backbone at a higher resolution (default efficientnet_b2 @ 288).

Usage (from the health-training folder):
    python day2/train_v2.py                                  # b2 @ 288, 14 epochs
    python day2/train_v2.py --model efficientnet_b0 --img 256 --epochs 14   # faster
    python day2/train_v2.py --model convnext_tiny --img 224 --lr 1e-4       # 2nd ensemble member
"""
import argparse
import copy
import csv
import math
import multiprocessing
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

import config as C
from common import (ImageTableDataset, build_model, eval_tf, image_dir_of, load_table,
                    select_device, train_tf)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_b2")
    p.add_argument("--img", type=int, default=288)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
    p.add_argument("--cw-power", type=float, default=0.25,
                   help="class weight = n_c^-power (0 = none, 1 = full inverse freq)")
    p.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-dir", default=C.TRAIN_DIR)
    p.add_argument("--eval-dir", default=C.EVAL_DIR)
    p.add_argument("--out", default=None, help="checkpoint path (default artifacts/<model>_<img>.pth)")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N images")
    p.add_argument("--no-pretrained", action="store_true", help="debug: random init")
    return p.parse_args()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y, _ in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    a = parse()
    device = select_device()
    print(f"[train] device={device} model={a.model} img={a.img}")
    out = a.out or str(C.ARTIFACT_DIR / f"{a.model}_{a.img}.pth")

    tr_df = load_table(a.train_dir)
    ev_df = load_table(a.eval_dir)
    if a.limit:
        tr_df = tr_df.sample(min(a.limit, len(tr_df)), random_state=0)
        ev_df = ev_df.sample(min(a.limit, len(ev_df)), random_state=0)
    print(f"[train] train={len(tr_df)}  eval={len(ev_df)}")

    tr_ds = ImageTableDataset(tr_df, image_dir_of(a.train_dir), train_tf(a.img))
    ev_ds = ImageTableDataset(ev_df, image_dir_of(a.eval_dir), eval_tf(a.img))
    kw = dict(num_workers=a.workers, pin_memory=False, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, drop_last=True, **kw)
    ev_dl = DataLoader(ev_ds, batch_size=a.bs * 2, shuffle=False, **kw)

    counts = np.array([(tr_df.dx == c).sum() for c in C.CLASS_NAMES], dtype=float)
    w = np.maximum(counts, 1) ** (-a.cw_power)
    w = w / (w * counts).sum() * counts.sum()           # mean weight per sample = 1
    print("[train] class weights:", dict(zip(C.CLASS_NAMES, np.round(w, 3))))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                    label_smoothing=a.ls)

    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)
    ema = EMA(model, a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(tr_dl)
    warm = int(a.warmup * len(tr_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.99 + 0.01)

    hist_path = out.replace(".pth", "_history.csv")
    best = -1.0
    with open(hist_path, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "acc", "macro_f1", "ema_acc", "ema_macro_f1", "secs"])
        for ep in range(1, a.epochs + 1):
            model.train()
            t0, run, n = time.time(), 0.0, 0
            for b, (x, y, _) in enumerate(tr_dl):
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                ema.update(model)
                run += loss.item() * len(y)
                n += len(y)
                if (b + 1) % 50 == 0:
                    print(f"   ep{ep} {b+1}/{len(tr_dl)} loss={run/n:.4f}", flush=True)

            acc, f1 = evaluate(model, ev_dl, device)
            eacc, ef1 = evaluate(ema.model, ev_dl, device)
            secs = time.time() - t0
            print(f"[ep {ep:02d}/{a.epochs}] loss={run/n:.4f} | raw acc={acc:.4f} f1={f1:.4f} "
                  f"| ema acc={eacc:.4f} f1={ef1:.4f} | {secs:.0f}s", flush=True)
            log.writerow([ep, round(run / n, 4), round(acc, 4), round(f1, 4),
                          round(eacc, 4), round(ef1, 4), round(secs)])
            fh.flush()

            for tag, m, ac, f in (("raw", model, acc, f1), ("ema", ema.model, eacc, ef1)):
                score = 0.5 * (ac + f)            # panel looks at Accuracy AND Macro-F1
                if score > best:
                    best = score
                    torch.save({"model_name": a.model, "img_size": a.img, "epoch": ep,
                                "weights": tag, "eval_acc": ac, "eval_macro_f1": f,
                                "class_names": C.CLASS_NAMES,
                                "model_state_dict": {k: v.cpu() for k, v in m.state_dict().items()}},
                               out)
                    print(f"    ✅ saved {tag} weights → {out}  (acc={ac:.4f}, f1={f:.4f})")
    print(f"[train] done. best (acc+f1)/2 = {best:.4f}. History → {hist_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
