"""
More teachers experiment: sweep K in {3, 5, 7}.

With K teachers:
  - Sensitivity per channel: Δ = 2/K
  - Noise at fixed ε: σ ∝ Δ = 2/K  (less noise with more teachers)
  - BUT each teacher trains on only N/K patients → may underfit

At K=7: Δ=2/7 ≈ 0.286 vs K=3: Δ=2/3 ≈ 0.667
→ sigma with K=7 is ~2.3× lower than K=3 at the same ε
→ should give meaningfully better utility especially at low ε

Key question: does the CANAL vs uniform gap change with K?
Hypothesis: if importance is flat (Gini≈0.26), the gap stays small regardless of K.

Conditions at each K: uniform vs CANAL, top 10% channels.
Budget split: rho_imp=10%, rho_rel=90%.

Usage:
  python canal_more_teachers.py --dataset isic --seeds 3 --te 60 --se 40
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
    ap.add_argument("--k_values", default="3,5,7", help="comma-separated K values")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    K_VALUES = [int(k) for k in args.k_values.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    print(f"[{args.dataset}] device={dev}  K_values={K_VALUES}  eps={EPS}  seeds={SEEDS}")
    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    N = len(train_ds)
    print(f"  train={N}  val={len(val_ds)}  in_ch={in_ch}")

    results = {
        "dataset": args.dataset, "K_values": K_VALUES, "ALPHA_IMP": ALPHA_IMP,
        "keep_frac": KEEP_FRAC, "epsilons": EPS, "seeds": SEEDS,
        "sweep": {},  # [K][eps] → {uniform, canal}
    }

    for K in K_VALUES:
        print(f"\n{'#'*72}\n#  K = {K}  (delta_per_channel = {2.0/K:.4f})\n{'#'*72}")
        teachers, caps_list = train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch)
        Cb = teachers[0].base * 4
        n_keep = max(1, int(round(KEEP_FRAC * Cb)))
        deltas = torch.full((Cb,), 2.0 / K, device=dev)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        importance = shared_importance(teachers, train_loader, dev).to(dev)
        imp_cpu = importance.cpu()
        sens_imp = 2.0 * CLIP_IMP / float(N)
        print(f"  C={Cb}  n_keep={n_keep}  importance max/mean={float(imp_cpu.max()/imp_cpu.mean()):.2f}x")

        results["sweep"][str(K)] = {}

        for eps in EPS:
            rho = eps_to_rho(eps)
            rho_imp = ALPHA_IMP * rho
            rho_rel = (1.0 - ALPHA_IMP) * rho
            noise_seed = int(eps * 100) + K * 17
            imp_noisy = add_dp_noise_to_importance(importance, sens_imp, rho_imp, seed=noise_seed)
            rank_desc = torch.argsort(imp_noisy, descending=True)
            top_indices = rank_desc[:n_keep]

            top_mask = torch.zeros(Cb, dtype=torch.bool, device=dev)
            top_mask[top_indices] = True

            sigma_uni = torch.zeros(Cb, device=dev)
            sigma_uni[top_mask] = correct_uniform_sigma(deltas[top_mask], rho_rel)

            sigma_canal = torch.zeros(Cb, device=dev)
            sigma_canal[top_mask] = correct_waterfilling_sigma(
                deltas[top_mask], imp_noisy[top_mask], rho_rel
            )

            print(f"\n  K={K}  eps={eps}  rho_rel={rho_rel:.4f}"
                  f"  sigma_uni={sigma_uni[top_mask].mean():.4f}"
                  f"  sigma_canal_mean={sigma_canal[top_mask].mean():.4f}")

            ep_res = {}
            for cond, sigma, mask in [("uniform", sigma_uni, top_mask),
                                       ("canal", sigma_canal, top_mask)]:
                print(f"    {cond} ...", end="", flush=True)
                t0 = time.time()
                cache = precompute_joint_cache(
                    teachers, caps_list, train_ds, sigma, mask, dev,
                    seed=int(eps * 1000) + K * 31 + (1 if cond == "uniform" else 2)
                )
                dices = []
                for s in SEEDS:
                    best, _ = train_student_distill(
                        train_ds, val_loader, cache, dev,
                        student_base=16, teacher_base=32,
                        n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                        seed=s, in_ch=in_ch,
                    )
                    dices.append(best)
                ep_res[cond] = cell_stats(dices)
                print(f"  Dice={ep_res[cond]['mean']:.4f}±{ep_res[cond]['sem']:.4f}  ({time.time()-t0:.1f}s)")

            ep_res["canal_minus_uniform"] = ep_res["canal"]["mean"] - ep_res["uniform"]["mean"]
            results["sweep"][str(K)][str(eps)] = ep_res

    # Summary: for each eps, show K progression
    for eps in EPS:
        print(f"\n{'='*72}")
        print(f"MORE TEACHERS  ({args.dataset.upper()}, eps={eps}, keep={KEEP_FRAC*100:.0f}%)")
        print(f"{'K':>4} | {'delta':>7} | {'uniform':>9} | {'canal':>9} | {'diff':>7}")
        print("-"*50)
        for K in K_VALUES:
            r = results["sweep"][str(K)][str(eps)]
            delta = 2.0 / K
            print(f"{K:>4} | {delta:>7.4f} | {r['uniform']['mean']:>9.4f} | "
                  f"{r['canal']['mean']:>9.4f} | {r['canal_minus_uniform']:>+7.4f}")

    out = HERE / "results" / f"{args.dataset}_more_teachers_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
