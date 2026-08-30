# -*- coding: utf-8 -*-
"""
Ablation (i): importance measure — Table 5 in paper.

Compares three conditions on ISIC 2018 at K=10, using clean (oracle)
importance with no honest-split overhead (full rho budget goes to release
noise), consistent with how Figure 3 and the honest-CANAL comparison are run.

Conditions:
  1. PATE + uniform       — full rho, uniform sigma across all channels
  2. PATE + CANAL (grad)  — full rho, water-filling with gradient-energy importance
  3. PATE + CANAL (act)   — full rho, water-filling with activation-norm importance

Reports Dice gain of each CANAL variant over uniform (positive = CANAL better),
matching Table 5 format in paper.

Usage:
  python isic_gradenergy_vs_actnorm.py
  python isic_gradenergy_vs_actnorm.py --K 10 --seeds 3 --epsilons "2,4,8" --out_tag "arxiv"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from isic_dataset import ISICDataset
from drive_pate_poc import (
    train_K_teachers, correct_uniform_sigma, precompute_pate_cache,
)
from drive_pate_pruning_joint import shared_importance, shared_importance_actnorm
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K",        type=int,   default=10)
    ap.add_argument("--seeds",    type=int,   default=3)
    ap.add_argument("--epsilons", type=str,   default="2,4,8")
    ap.add_argument("--te",       type=int,   default=50, help="teacher epochs")
    ap.add_argument("--se",       type=int,   default=40, help="student epochs")
    ap.add_argument("--size",     type=int,   default=96)
    ap.add_argument("--out_tag",  type=str,   default="")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS   = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds = ISICDataset(split="train", size=args.size)
    val_ds   = ISICDataset(split="val",   size=args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[ISIC grad-energy vs act-norm] K={args.K}  eps={EPS}  seeds={SEEDS}  device={dev}")

    # ---- train K teachers ----
    print(f"\nTraining K={args.K} teachers ({args.te} epochs)...")
    t0 = time.time()
    teachers, caps_list = train_K_teachers(
        train_ds, K=args.K, device=dev, n_epochs=args.te, in_ch=3,
    )
    print(f"  done in {time.time()-t0:.0f}s")

    Cb     = teachers[0].base * 4
    deltas = torch.full((Cb,), 2.0 / args.K, device=dev)
    print(f"  bottleneck C={Cb}")

    # ---- compute importances (oracle — no DP cost) ----
    print("Computing importances...")
    imp_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    imp_grad = shared_importance(teachers, imp_loader, dev).to(dev)
    imp_act  = shared_importance_actnorm(teachers, imp_loader, dev).to(dev)
    R_grad = float(imp_grad.max() / imp_grad.min())
    R_act  = float(imp_act.max()  / imp_act.min())
    print(f"  R(grad-energy)={R_grad:.2f}  R(act-norm)={R_act:.2f}")

    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    save_path  = HERE / "results" / f"isic_gradenergy_vs_actnorm{tag_suffix}_results.json"
    results = {
        "dataset": "isic", "K": args.K,
        "epsilons": EPS, "seeds": SEEDS,
        "te": args.te, "se": args.se,
        "importance": {"R_grad": R_grad, "R_act": R_act},
        "series": {"uniform": {}, "CANAL_grad": {}, "CANAL_act": {}},
    }

    def run_cell(sigma, label, eps):
        cache = precompute_pate_cache(
            teachers, caps_list, train_ds, sigma, dev,
            seed=42 + int(eps * 10),
        )
        dices = []
        for s in SEEDS:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, dev,
                student_base=16, teacher_base=32,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                seed=s, in_ch=3,
            )
            dices.append(best)
            print(f"    {label} eps={eps} seed={s}: {best:.4f}  ({time.time()-t0:.0f}s)")
        return {
            "dices": dices,
            "mean":  float(np.mean(dices)),
            "std":   float(np.std(dices)),
            "sem":   float(np.std(dices) / np.sqrt(len(dices))),
        }

    for eps in EPS:
        rho = eps_to_rho(eps)
        sigma_uni  = correct_uniform_sigma(deltas, rho)
        sigma_grad = correct_waterfilling_sigma(deltas, imp_grad, rho)
        sigma_act  = correct_waterfilling_sigma(deltas, imp_act,  rho)
        print(f"\n{'='*60}\neps={eps}  rho={rho:.5f}")

        print("  [uniform]    ...")
        results["series"]["uniform"][str(eps)]    = run_cell(sigma_uni,  "uniform",    eps)

        print("  [CANAL grad] ...")
        results["series"]["CANAL_grad"][str(eps)] = run_cell(sigma_grad, "CANAL_grad", eps)

        print("  [CANAL act]  ...")
        results["series"]["CANAL_act"][str(eps)]  = run_cell(sigma_act,  "CANAL_act",  eps)

        save_path.write_text(json.dumps(results, indent=2))

    # ---- summary table (gain over uniform, matching paper Table 5 format) ----
    print(f"\n{'='*60}")
    print(f"Table 5 — Dice gain of CANAL over uniform  (K={args.K}, ISIC)")
    print(f"  {'Importance':<20s}", "  ".join(f"eps={e}" for e in EPS))
    for method, label in [("CANAL_grad", "Gradient energy"), ("CANAL_act", "Activation norm")]:
        gains = [results["series"][method][str(e)]["mean"]
                 - results["series"]["uniform"][str(e)]["mean"] for e in EPS]
        print(f"  {label:<20s}", "  ".join(f"{g:+.4f}" for g in gains))

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
