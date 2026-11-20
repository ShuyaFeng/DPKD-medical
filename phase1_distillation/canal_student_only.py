# -*- coding: utf-8 -*-
"""
canal_student_only.py — Student-only baseline for Table 1.

Trains the same student architecture used in CANAL (TinyUNet base=16) with
segmentation loss only, NO distillation, NO DP noise. Provides the lower-bound
row ("Student only") in Table 1.

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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, train_teacher, evaluate_metrics

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
        torch.manual_seed(s)
        student = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(dev)
        loader  = DataLoader(train_ds, batch_size=4, shuffle=True)
        train_teacher(student, loader, n_epochs=args.se, lr=1e-3, device=dev,
                      val_loader=val_loader)
        metrics = evaluate_metrics(student, val_loader, dev)
        dices.append(metrics["vessel_dice"])
        print(f"  seed={s}  Dice={metrics['vessel_dice']:.4f}")

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
