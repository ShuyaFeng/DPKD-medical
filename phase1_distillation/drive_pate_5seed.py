"""
M1 PATE multi-teacher: 5-seed confirmation.

Identical setup to drive_pate_poc.py but with student_seeds = [100, 200,
300, 400, 500]. Confirms or refutes the 3-seed finding that PATE K=5
gives +0.03 paired Dice lift over K=1 baseline at ε=8 and ε=16.
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
    N_train = len(train_ds)

    K_values      = [1, 3, 5]
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    base_T        = 32
    Cb_T          = base_T * 4
    n_seeds       = len(student_seeds)

    results = {
        "K_values": K_values, "epsilons": epsilons,
        "student_seeds": student_seeds, "sweep": {},
    }

    for K in K_values:
        partitions = partition_dataset(N_train, K)
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
            print(f"\n  ε={eps:>4.1f}  ρ={rho:.4f}  σ_uniform={sigma[0].item():.4f}")

            cache = precompute_pate_cache(teachers, caps_list, train_ds,
                                           sigma, device,
                                           seed=42 + int(eps * 10))

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
            sem_d  = std_d / np.sqrt(n_seeds)
            print(f"   → K={K} ε={eps}:  {mean_d:.4f} ± {std_d:.4f}")
            results["sweep"][K]["epsilons"][eps] = {
                "dices": dices, "mean": mean_d, "std": std_d, "sem": sem_d,
                "sigma": float(sigma[0]),
            }

    # --- summary ---
    print("\n" + "=" * 110)
    print(f"{'K':>3}  {'ε':>5}  {'σ_uni':>10}  {'Dice mean ± std':>22}  "
          f"{'paired vs K=1':>22}  {'sig':>10}")
    print("-" * 110)
    for K in K_values:
        for eps in epsilons:
            r = results["sweep"][K]["epsilons"][eps]
            mean_d, std_d, sigma = r["mean"], r["std"], r["sigma"]
            if K == 1:
                paired_str, sig = "", ""
            else:
                base_dices = results["sweep"][1]["epsilons"][eps]["dices"]
                paired = [k - b for k, b in zip(r["dices"], base_dices)]
                pm  = float(np.mean(paired))
                psd = float(np.std(paired))
                psem = psd / np.sqrt(len(paired))
                tstat = pm / psem if psem > 0 else 0
                if   abs(tstat) >= 3:  sig = f"*** {tstat:+.1f}σ"
                elif abs(tstat) >= 2:  sig = f"**  {tstat:+.1f}σ"
                elif abs(tstat) >= 1:  sig = f"~   {tstat:+.1f}σ"
                else:                  sig = f"≈   {tstat:+.1f}σ"
                paired_str = f"{pm:+.4f}±{psem:.4f}"
                results["sweep"][K]["epsilons"][eps]["paired_mean"] = pm
                results["sweep"][K]["epsilons"][eps]["paired_sem"]  = psem
                results["sweep"][K]["epsilons"][eps]["t_stat"]      = tstat
            print(f"{K:>3}  {eps:>5.1f}  {sigma:>10.4f}  {mean_d:.4f} ± {std_d:.4f}  "
                  f"{paired_str:>22}  {sig:>10}")
    print("=" * 110)

    out_path = Path(__file__).parent / "drive_pate_5seed_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {1: "#d62728", 3: "#ff7f0e", 5: "#2ca02c"}

    # a: Dice vs eps per K
    ax = axes[0]
    for K in K_values:
        means = [results["sweep"][K]["epsilons"][e]["mean"] for e in epsilons]
        stds  = [results["sweep"][K]["epsilons"][e]["std"]  for e in epsilons]
        ax.errorbar(epsilons, means, yerr=stds, fmt="o-", color=colors[K],
                    lw=2, ms=10, capsize=6,
                    label=f"K={K}  (Δ=2/{K}={2/K:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title(f"(a) PATE Dice vs ε ({n_seeds} seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    # b: paired lift with per-seed dots
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    for K in [3, 5]:
        means_l, sems_l = [], []
        for eps in epsilons:
            base = results["sweep"][1]["epsilons"][eps]["dices"]
            this = results["sweep"][K]["epsilons"][eps]["dices"]
            paired = [k - b for k, b in zip(this, base)]
            means_l.append(np.mean(paired))
            sems_l.append(np.std(paired) / np.sqrt(len(paired)))
            for p in paired:
                ax.scatter(eps, p, color=colors[K], alpha=0.4, s=40, zorder=3)
        ax.errorbar(epsilons, means_l, yerr=sems_l, fmt="D-", color=colors[K],
                    lw=2.5, ms=11, capsize=6, label=f"K={K} − K=1")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15, label="±0.002 noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel(f"Paired Dice lift vs K=1  (dots = {n_seeds} seeds)")
    ax.set_title(f"(b) Lift over K=1 baseline (paired, {n_seeds} seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"M1 PATE 5-seed confirmation — honest multi-teacher feature aggregation",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / "drive_pate_5seed.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
