# -*- coding: utf-8 -*-
"""
canal_student_only.py — Student-only baseline for Table 1.

Trains the same student architecture used in CANAL (TinyUNet base=16) with
segmentation loss only, NO distillation, NO DP noise. Tracks best val Dice
across epochs (same as train_student_distill) for a fair comparison.

Usage:
  python canal_student_only.py --dataset isic   --seeds 5 --se 40
  python canal_student_only.py --dataset kvasir --seeds 5 --se 40
  python canal_student_only.py --dataset busi   --seeds 5 --se 40
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, evaluate_metrics
from synthetic_demo import task_loss

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def get_dataset(name, split, size):
    if name == "isic":
        from isic_dataset import ISICDataset
        return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset
        return KvasirDataset(split, size), 3
    if name == "busi":
        from busi_dataset import BUSIDataset
        return BUSIDataset(split, size), 1
    raise ValueError(name)


def train_student_seg_only(train_ds, val_loader, device, in_ch, n_epochs, lr, seed):
    """Train TinyUNet(base=16) with segmentation loss only; return best val Dice."""
    torch.manual_seed(seed)
    student = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    best_dice = 0.0
    for ep in range(n_epochs):
        student.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = task_loss(student(x), y)
            loss.backward()
            opt.step()
            total += loss.item()

        metrics = evaluate_metrics(student, val_loader, device)
        val_dice = metrics["vessel_dice"]
        if val_dice > best_dice:
            best_dice = val_dice

        if (ep + 1) % 5 == 0 or ep == 0 or ep == n_epochs - 1:
            print(f"    ep {ep+1:3d}/{n_epochs}  loss={total/len(loader):.4f}"
                  f"  val={val_dice:.4f}  best={best_dice:.4f}")

    return best_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size",    type=int, default=96)
    ap.add_argument("--seeds",   type=int, default=5)
    ap.add_argument("--se",      type=int, default=40, help="student training epochs")
    ap.add_argument("--suffix",  default="")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))
    sfx   = f"_{args.suffix}" if args.suffix else ""

    print(f"[{args.dataset}] device={dev}  seeds={SEEDS}  se={args.se}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds,   _     = get_dataset(args.dataset, "val",   args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  in_ch={in_ch}")

    dices = []
    for s in SEEDS:
        print(f"\n  seed={s} ...", flush=True)
        best = train_student_seg_only(
            train_ds, val_loader, dev, in_ch,
            n_epochs=args.se, lr=1e-3, seed=s,
        )
        dices.append(best)
        print(f"  seed={s}  best_dice={best:.4f}")

    mean = float(np.mean(dices))
    std  = float(np.std(dices))
    sem  = std / math.sqrt(len(dices))
    print(f"\n[{args.dataset}] student-only  mean={mean:.4f}  std={std:.4f}  sem={sem:.4f}")

    out = HERE / "results" / f"{args.dataset}_student_only_results{sfx}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset, "method": "student_only",
        "seeds": SEEDS, "dices": dices,
        "mean": mean, "std": std, "sem": sem,
    }, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
