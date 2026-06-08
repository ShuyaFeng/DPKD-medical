"""
K saturation sweep: PATE multi-teacher Dice vs K ∈ {1, 3, 5, 8, 10}.

Tests whether the +0.031 Dice lift at K=5 saturates or continues to grow.
DRIVE N=20 means:
  K=1:  20 patients per teacher
  K=3:  6-7 patients per teacher
  K=5:  4 patients per teacher
  K=8:  2-3 patients per teacher
  K=10: 2 patients per teacher

Each K cell: 5 seeds × 3 ε. Same training procedure as drive_pate_poc.py
(teachers seeded with 1000+k, students with 100, 200, 300, 400, 500).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import DriveDataset
from synthetic_demo import eps_to_rho
from drive_student_distill import train_student_distill
from drive_pate_poc import (
    correct_uniform_sigma, partition_dataset, train_K_teachers,
    precompute_pate_cache,
)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    K_values      = [1, 3, 5, 8, 10]
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32
    Cb_T          = base_T * 4

    results = {"K_values": K_values, "epsilons": epsilons,
               "student_seeds": student_seeds, "sweep": {}}

    for K in K_values:
        partitions = partition_dataset(len(train_ds), K)
        sizes = [len(p) for p in partitions]
        print(f"\n{'='*72}")
        print(f"K = {K}   partition sizes = {sizes}   Δ = 2/K = {2/K:.3f}")
        print(f"{'='*72}")

        teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)

        deltas = torch.full((Cb_T,), 2.0 / K, device=device)
        results["sweep"][K] = {"partition_sizes": sizes, "epsilons": {}}

        for eps in epsilons:
            rho   = eps_to_rho(eps)
            sigma = correct_uniform_sigma(deltas, rho)
            print(f"\n  ε={eps:>4.1f}  σ_uniform={sigma[0].item():.4f}")

            cache = precompute_pate_cache(teachers, caps_list, train_ds, sigma,
                                          device, seed=42 + int(eps * 10))
            dices = []
            for s in student_seeds:
                t0 = time.time()
                best, _ = train_student_distill(
                    train_ds, val_loader, cache, device,
                    student_base=16, teacher_base=base_T,
                    n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                )
                dices.append(best)
                print(f"    seed={s}: {best:.4f}  ({time.time()-t0:.1f}s)")
            mean_d = float(np.mean(dices))
            std_d  = float(np.std(dices))
            sem_d  = std_d / np.sqrt(len(dices))
            print(f"   → K={K} ε={eps}:  {mean_d:.4f} ± {std_d:.4f}")
            results["sweep"][K]["epsilons"][eps] = {
                "dices": dices, "mean": mean_d, "std": std_d, "sem": sem_d,
                "sigma": float(sigma[0]),
            }

    # Summary
    print("\n" + "=" * 110)
    print(f"{'K':>3}  {'ε':>5}  {'σ':>10}  {'Dice mean ± std':>22}  "
          f"{'paired vs K=1':>22}  {'sig':>12}")
    print("-" * 110)
    for K in K_values:
        for eps in epsilons:
            r = results["sweep"][K]["epsilons"][eps]
            mean_d, std_d, sigma = r["mean"], r["std"], r["sigma"]
            if K == 1:
                paired_str, sig = "", ""
            else:
                base = results["sweep"][1]["epsilons"][eps]["dices"]
                paired = [k - b for k, b in zip(r["dices"], base)]
                pm  = float(np.mean(paired))
                psem = float(np.std(paired) / np.sqrt(len(paired)))
                tstat = pm / psem if psem > 0 else 0
                if   abs(tstat) >= 3: sig = f"*** {tstat:+.1f}σ"
                elif abs(tstat) >= 2: sig = f"**  {tstat:+.1f}σ"
                elif abs(tstat) >= 1: sig = f"~   {tstat:+.1f}σ"
                else:                 sig = f"≈   {tstat:+.1f}σ"
                paired_str = f"{pm:+.4f}±{psem:.4f}"
            print(f"{K:>3}  {eps:>5.1f}  {sigma:>10.4f}  "
                  f"{mean_d:.4f} ± {std_d:.4f}  {paired_str:>22}  {sig:>12}")
    print("=" * 110)

    out_path = Path(__file__).parent / "drive_pate_K_saturation_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    eps_colors = {2.0: "#d62728", 8.0: "#1f77b4", 16.0: "#2ca02c"}

    # Panel a: Dice vs K per ε
    ax = axes[0]
    for eps in epsilons:
        means = [results["sweep"][K]["epsilons"][eps]["mean"] for K in K_values]
        stds  = [results["sweep"][K]["epsilons"][eps]["std"]  for K in K_values]
        ax.errorbar(K_values, means, yerr=stds, fmt="o-", color=eps_colors[eps],
                    lw=2, ms=10, capsize=6, label=f"ε={int(eps)}")
    ax.set_xticks(K_values)
    ax.set_xlabel("K  (number of teachers)")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) Dice vs K — saturation curve")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)

    # Panel b: paired lift vs K=1
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    for eps in epsilons:
        lifts_m, lifts_sem = [], []
        for K in K_values:
            base_d = results["sweep"][1]["epsilons"][eps]["dices"]
            k_d    = results["sweep"][K]["epsilons"][eps]["dices"]
            paired = [k - b for k, b in zip(k_d, base_d)]
            lifts_m.append(float(np.mean(paired)))
            lifts_sem.append(float(np.std(paired) / np.sqrt(len(paired))))
        ax.errorbar(K_values, lifts_m, yerr=lifts_sem, fmt="o-",
                    color=eps_colors[eps], lw=2, ms=10, capsize=6,
                    label=f"ε={int(eps)}")
    ax.set_xticks(K_values)
    ax.set_xlabel("K  (number of teachers)")
    ax.set_ylabel("paired Dice lift vs K=1 (mean ± SEM)")
    ax.set_title("(b) Lift over K=1 — does it keep climbing?")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("M1 PATE — K saturation sweep on real DRIVE (5 seeds per cell)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / "drive_pate_K_saturation.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
