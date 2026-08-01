"""
Feature loss weight (lambda_feat) sweep.

Tests whether higher teacher signal (larger lambda) helps the student learn
better from DP-noised bottleneck features, and whether it exposes CANAL gains.

Fixed config: top 10% channels, K=3, seeds=3, all 3 datasets.
Two allocation conditions at each lambda:
  - top_uniform : top-10% channels, uniform sigma
  - top_canal   : top-10% channels, WF sigma

Lambda range covers roughly 10% to 91% feature loss fraction of total loss:
  lambda_feat   feature_frac ≈ lambda / (1 + lambda)
  ----------    --------------------------------
  0.1           9%   (task strongly dominates — barely any teacher signal)
  0.4           29%  (current paper default)
  1.0           50%  (equal task and feature)
  2.0           67%  (feature moderately dominates)
  5.0           83%  (feature strongly dominates)
  10.0          91%  (feature very strongly dominates)

Budget split: rho_imp=10%, rho_rel=90% (same as selection ablation).

Usage:
  python canal_lambda_sweep.py --dataset isic --seeds 3 --te 60 --se 40
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_pate_poc import train_K_teachers, correct_uniform_sigma
from drive_pate_pruning_joint import shared_importance, precompute_joint_cache
from drive_pate_canal_combined import add_dp_noise_to_importance, correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

ALPHA_IMP = 0.1
CLIP_IMP = 1e-8
K = 3
KEEP_FRAC = 0.1


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


def run_students(train_ds, val_loader, teachers, caps_list, active_mask, sigma,
                 device, seeds, n_epochs_student, lambda_feat, in_ch, cache_seed):
    cache = precompute_joint_cache(
        teachers, caps_list, train_ds, sigma, active_mask, device, seed=cache_seed
    )
    dices = []
    for s in seeds:
        best, _ = train_student_distill(
            train_ds, val_loader, cache, device,
            student_base=16, teacher_base=32,
            n_epochs=n_epochs_student, lr=1e-3, lambda_feat=lambda_feat,
            seed=s, in_ch=in_ch,
        )
        dices.append(best)
    return dices


def cell_stats(dices):
    m = float(np.mean(dices))
    s = float(np.std(dices))
    sem = s / math.sqrt(len(dices))
    return {"dices": dices, "mean": m, "std": s, "sem": sem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isic", choices=["isic", "kvasir", "busi"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--te", type=int, default=60)
    ap.add_argument("--se", type=int, default=40)
    ap.add_argument("--epsilons", default="0.5,1,2,4,6")
    ap.add_argument("--lambdas", default="0.1,0.4,1.0,2.0,5.0,10.0",
                    help="comma-separated lambda_feat values")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    LAMBDAS = [float(l) for l in args.lambdas.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K={K}  eps={EPS}")
    print(f"  lambdas={LAMBDAS}  seeds={SEEDS}")

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
    Cb = teachers[0].base * 4
    n_keep = max(1, int(round(KEEP_FRAC * Cb)))
    print(f"  bottleneck C={Cb}  n_keep={n_keep} ({KEEP_FRAC*100:.0f}%)")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    importance = shared_importance(teachers, train_loader, dev).to(dev)
    sens_imp = 2.0 * CLIP_IMP / float(N)
    deltas = torch.full((Cb,), 2.0 / K, device=dev)

    results = {
        "dataset": args.dataset, "K": K, "keep_frac": KEEP_FRAC, "n_keep": n_keep,
        "epsilons": EPS, "lambdas": LAMBDAS, "seeds": SEEDS, "ALPHA_IMP": ALPHA_IMP,
        "C": Cb,
        "sweep": {},  # [eps][lambda] → {top_uniform: cell, top_canal: cell}
    }

    for eps in EPS:
        rho = eps_to_rho(eps)
        rho_imp = ALPHA_IMP * rho
        rho_rel = (1.0 - ALPHA_IMP) * rho
        noise_seed = int(eps * 100) + 7
        imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
        rank_desc = torch.argsort(imp_noisy, descending=True)
        top_indices = rank_desc[:n_keep]

        # Build active mask + sigma for both allocation modes (reused across all lambdas)
        top_mask_uni = torch.zeros(Cb, dtype=torch.bool, device=dev)
        top_mask_uni[top_indices] = True
        sigma_uni = torch.zeros(Cb, device=dev)
        sigma_uni[top_mask_uni] = correct_uniform_sigma(deltas[top_mask_uni], rho_rel)

        top_mask_canal = torch.zeros(Cb, dtype=torch.bool, device=dev)
        top_mask_canal[top_indices] = True
        sigma_canal = torch.zeros(Cb, device=dev)
        sigma_canal[top_mask_canal] = correct_waterfilling_sigma(
            deltas[top_mask_canal], imp_noisy[top_mask_canal], rho_rel
        )

        print(f"\n{'='*64}\neps={eps}  rho_rel={rho_rel:.4f}  sigma_uni[active]={sigma_uni[top_mask_uni].mean():.4f}"
              f"  sigma_canal[active]={sigma_canal[top_mask_canal].mean():.4f}")
        results["sweep"][str(eps)] = {}

        for lam in LAMBDAS:
            feat_frac = lam / (1.0 + lam)
            print(f"\n  lambda={lam:.1f}  (feature_frac≈{feat_frac*100:.0f}%)")

            # uniform with this lambda
            cache_seed_uni = int(eps * 1000) + int(lam * 100) + 10
            t0 = time.time()
            dices_uni = run_students(train_ds, val_loader, teachers, caps_list,
                                     top_mask_uni, sigma_uni, dev, SEEDS, args.se, lam, in_ch,
                                     cache_seed_uni)
            cell_uni = cell_stats(dices_uni)

            # canal with this lambda
            cache_seed_canal = int(eps * 1000) + int(lam * 100) + 20
            dices_canal = run_students(train_ds, val_loader, teachers, caps_list,
                                       top_mask_canal, sigma_canal, dev, SEEDS, args.se, lam, in_ch,
                                       cache_seed_canal)
            cell_canal = cell_stats(dices_canal)
            elapsed = time.time() - t0

            results["sweep"][str(eps)][str(lam)] = {
                "lambda_feat": lam,
                "feature_frac": feat_frac,
                "top_uniform": cell_uni,
                "top_canal": cell_canal,
                "canal_minus_uniform": cell_canal["mean"] - cell_uni["mean"],
            }
            print(f"    top_uni={cell_uni['mean']:.4f}  top_canal={cell_canal['mean']:.4f}"
                  f"  diff={cell_canal['mean']-cell_uni['mean']:+.4f}  ({elapsed:.1f}s)")

    # Summary table: for each eps, show uniform and canal across all lambdas
    for eps in EPS:
        print(f"\n{'='*72}")
        print(f"LAMBDA SWEEP  ({args.dataset.upper()}, eps={eps}, K={K}, keep={KEEP_FRAC*100:.0f}%)")
        print(f"{'lambda':>7} | {'feat%':>6} | {'top_uni':>9} | {'top_canal':>10} | {'canal-uni':>10}")
        print("-"*72)
        for lam in LAMBDAS:
            r = results["sweep"][str(eps)][str(lam)]
            print(f"{lam:>7.1f} | {r['feature_frac']*100:>5.0f}% | "
                  f"{r['top_uniform']['mean']:>9.4f} | {r['top_canal']['mean']:>10.4f} | "
                  f"{r['canal_minus_uniform']:>+10.4f}")

    out = HERE / "results" / f"{args.dataset}_lambda_sweep_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
