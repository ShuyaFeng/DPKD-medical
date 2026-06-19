"""
PATE + CANAL combined: does importance-weighted allocation help on top of
multi-teacher sensitivity reduction?

Predicted outcome from the R^{1/4} bound (Corollary 1): the CANAL σ ratio
between channels is (s_max/s_min)^{1/4}, INDEPENDENT of K. So if R≈2 on
DRIVE → σ ratio ≈1.19 → Dice gap ≈0 just like single-teacher CANAL.

We verify here at K=5 (paper's headline K), paired 5 seeds × 3 ε.

Each teacher's importance s^(k) is computed on its own training cohort
(the same disjoint partition used for teacher training). Aggregate
importance s̄ = mean(s^(k)) used in CANAL's waterfilling formula with
Δ = 2/K.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps,
)
from synthetic_demo import eps_to_rho
from drive_student_distill import train_student_distill
from drive_pate_poc import (
    correct_uniform_sigma, partition_dataset, train_K_teachers,
    precompute_pate_cache,
)


def correct_waterfilling_sigma(deltas, importance, rho):
    """WF sigma with the correct factor-of-2 in zCDP."""
    s = importance.clamp(min=1e-12)
    kappa = ((deltas * s.sqrt()).sum() / (2.0 * rho)).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def compute_teacher_importance(teacher, subset_indices, train_ds, device):
    """Compute per-channel importance for teacher_k on its own cohort.
    Temporarily enables grad on the (post-train, frozen) teacher params."""
    subset = Subset(train_ds, subset_indices)
    bs = min(4, len(subset))
    sub_loader = DataLoader(subset, batch_size=bs, shuffle=False)
    # Teachers were frozen in train_K_teachers; re-enable for grad computation.
    for p in teacher.parameters():
        p.requires_grad_(True)
    try:
        result = compute_importance(teacher, sub_loader, device)
    finally:
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
    return result


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    K              = 5
    epsilons       = [1.0, 2.0, 3.0, 4.0, 5.0]
    student_seeds  = [100, 200, 300, 400, 500]
    base_T         = 32
    Cb_T           = base_T * 4

    print(f"\n[1/3] Training K={K} teachers on disjoint cohorts...")
    teachers, caps_list = train_K_teachers(train_ds, K, device, n_epochs=60)
    partitions = partition_dataset(len(train_ds), K)

    print(f"\n[2/3] Computing per-teacher importance + aggregate...")
    importance_per_teacher = []
    for k, teacher in enumerate(teachers):
        imp = compute_teacher_importance(teacher, partitions[k], train_ds, device).to(device)
        importance_per_teacher.append(imp)
        ratio_k = (imp.max() / imp.min()).item()
        print(f"  teacher {k+1}: imp range [{imp.min().item():.3e}, "
              f"{imp.max().item():.3e}]  ratio={ratio_k:.2f}×")

    imp_agg = torch.stack(importance_per_teacher).mean(dim=0)
    R_agg = (imp_agg.max() / imp_agg.min()).item()
    print(f"\n  aggregate importance ratio R = {R_agg:.2f}×")
    print(f"  predicted CANAL σ ratio R^(1/4) = {R_agg**0.25:.3f}×")

    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    print(f"\n[3/3] PATE+uniform vs PATE+CANAL × 5 seeds × 3 ε ...")
    results = {"K": K, "R_aggregate": R_agg,
               "epsilons": epsilons, "student_seeds": student_seeds, "sweep": {}}

    for eps in epsilons:
        rho       = eps_to_rho(eps)
        sigma_uni = correct_uniform_sigma(deltas, rho)
        sigma_wf  = correct_waterfilling_sigma(deltas, imp_agg, rho)
        top10 = torch.argsort(imp_agg, descending=True)[:10]
        bot10 = torch.argsort(imp_agg, descending=False)[:10]
        sigma_ratio_actual = (sigma_wf[bot10].mean() / sigma_wf[top10].mean()).item()
        print(f"\n  ε={eps:>4.1f}  ρ={rho:.4f}")
        print(f"     σ_uniform = {sigma_uni[0].item():.4f}")
        print(f"     σ_CANAL:  top10 mean = {sigma_wf[top10].mean().item():.4f}  "
              f"bot10 mean = {sigma_wf[bot10].mean().item():.4f}  "
              f"actual ratio = {sigma_ratio_actual:.3f}×")

        results["sweep"][eps] = {}
        for mech_name, sigma in [("uniform", sigma_uni), ("CANAL", sigma_wf)]:
            print(f"\n     --- PATE + {mech_name} ---")
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
                print(f"       seed={s}: {best:.4f}  ({time.time()-t0:.1f}s)")
            mean_d = float(np.mean(dices))
            std_d  = float(np.std(dices))
            print(f"     → PATE+{mech_name} ε={eps}:  {mean_d:.4f} ± {std_d:.4f}")
            results["sweep"][eps][mech_name] = {
                "dices": dices, "mean": mean_d, "std": std_d,
                "sigma_top10": float(sigma[top10].mean()),
                "sigma_bot10": float(sigma[bot10].mean()),
            }

    # ---- summary ----
    print("\n" + "=" * 110)
    print(f"K = {K},  aggregate R = {R_agg:.2f}×,  predicted σ ratio = {R_agg**0.25:.3f}×")
    print("-" * 110)
    print(f"{'ε':>5}  {'PATE+uniform':>22}  {'PATE+CANAL':>22}  {'paired lift':>22}  {'sig':>14}")
    print("-" * 110)
    for eps in epsilons:
        u_d = results["sweep"][eps]["uniform"]["dices"]
        c_d = results["sweep"][eps]["CANAL"]["dices"]
        u_mean = float(np.mean(u_d)); u_std = float(np.std(u_d))
        c_mean = float(np.mean(c_d)); c_std = float(np.std(c_d))
        paired = [c - u for u, c in zip(u_d, c_d)]
        pm     = float(np.mean(paired))
        psem   = float(np.std(paired) / np.sqrt(len(paired)))
        tstat  = pm / psem if psem > 0 else 0
        if   abs(tstat) >= 3: sig = f"*** {tstat:+.1f}σ"
        elif abs(tstat) >= 2: sig = f"**  {tstat:+.1f}σ"
        elif abs(tstat) >= 1: sig = f"~   {tstat:+.1f}σ"
        else:                 sig = f"≈   {tstat:+.1f}σ"

        results["sweep"][eps]["paired_lift_mean"] = pm
        results["sweep"][eps]["paired_lift_sem"]  = psem
        results["sweep"][eps]["t_stat"]           = tstat

        print(f"{eps:>5.1f}  {u_mean:.4f} ± {u_std:.4f}  "
              f"{c_mean:.4f} ± {c_std:.4f}  "
              f"{pm:+.4f} ± {psem:.4f}  {sig:>14}")
    print("=" * 110)

    out_path = Path(__file__).parent / "drive_pate_canal_combined_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    u_means = [results["sweep"][e]["uniform"]["mean"] for e in epsilons]
    u_stds  = [results["sweep"][e]["uniform"]["std"]  for e in epsilons]
    c_means = [results["sweep"][e]["CANAL"]["mean"]   for e in epsilons]
    c_stds  = [results["sweep"][e]["CANAL"]["std"]    for e in epsilons]
    ax.errorbar(epsilons, u_means, yerr=u_stds, fmt="s-", color="#2ca02c",
                lw=2.5, ms=10, capsize=6, label=f"PATE K={K} + uniform")
    ax.errorbar(epsilons, c_means, yerr=c_stds, fmt="^-", color="#9467bd",
                lw=2.5, ms=10, capsize=6, label=f"PATE K={K} + CANAL")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title(f"(a) PATE+uniform vs PATE+CANAL at K={K} (5 seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    lifts_m, lifts_sem = [], []
    for eps in epsilons:
        u_d = results["sweep"][eps]["uniform"]["dices"]
        c_d = results["sweep"][eps]["CANAL"]["dices"]
        paired = [c - u for u, c in zip(u_d, c_d)]
        lifts_m.append(float(np.mean(paired)))
        lifts_sem.append(float(np.std(paired) / np.sqrt(len(paired))))
        for v in paired:
            ax.scatter(eps, v, color="#9467bd", alpha=0.45, s=45, zorder=3)
    ax.errorbar(epsilons, lifts_m, yerr=lifts_sem, fmt="D-", color="#9467bd",
                lw=2.5, ms=11, capsize=6, label="paired CANAL − uniform (per seed)")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15, label="±0.002 noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("paired Dice lift  (CANAL − uniform)")
    ax.set_title(f"(b) Does CANAL allocation matter on top of PATE? (paired, K={K})")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"PATE K={K} + CANAL test  —  R^(1/4) = {R_agg**0.25:.2f}× independent of K  "
        f"(predicted null on RGB DRIVE)",
        fontweight="bold", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(__file__).parent / "drive_pate_canal_combined.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
