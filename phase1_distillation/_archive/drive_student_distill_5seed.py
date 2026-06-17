"""
5-seed student distillation: WF vs uniform (paired test).

Confirms or refutes the 3-seed finding that channel-WF beats uniform with
statistical significance at ε=2 and ε=8 on real DRIVE.

Same setup as drive_student_distill_multiseed.py but with student_seeds =
[100, 200, 300, 400, 500] and 5 baseline seeds for fair reference.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps,
)
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
)
from drive_student_distill import precompute_cache, train_student_distill


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    student_seeds = [100, 200, 300, 400, 500]
    n_seeds = len(student_seeds)

    print(f"\n[1/3] Training teacher...")
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    t_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    train_teacher(teacher, t_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    importance = compute_importance(teacher, t_loader, device).to(device)
    caps       = collect_caps(teacher, t_loader, device).to(device)
    Cb_T = importance.shape[0]
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    print(f"  teacher clean Dice = {clean_dice:.4f}")

    print(f"\n[2/3] Student baselines ({n_seeds} seeds)...")
    baseline_dices = []
    for s in student_seeds:
        torch.manual_seed(s)
        sb = TinyUNet(in_ch=3, num_classes=2, base=16).to(device)
        sb_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        train_teacher(sb, sb_loader, n_epochs=60, lr=1e-3, device=device)
        d = evaluate_vessel_dice(sb, val_loader, device)
        baseline_dices.append(d)
        print(f"  seed={s}: {d:.4f}")
    baseline_mean = float(np.mean(baseline_dices))
    baseline_std  = float(np.std(baseline_dices))
    print(f"  baseline = {baseline_mean:.4f} ± {baseline_std:.4f}")

    print(f"\n[3/3] Student distillation: WF vs uniform × {n_seeds} seeds...")
    epsilons = [2.0, 8.0, 16.0]
    K = 10
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    results = {
        "clean_teacher":          clean_dice,
        "student_baseline_mean":  baseline_mean,
        "student_baseline_std":   baseline_std,
        "student_baseline_dices": baseline_dices,
        "K": K, "lambda_feat": 0.4, "n_epochs": 40,
        "student_seeds": student_seeds,
        "sweep": {},
    }

    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma_uni = uniform_sigma(deltas, rho)
        sigma_wf  = waterfilling_sigma(deltas, importance, rho)
        results["sweep"][eps] = {}

        cache_uni = precompute_cache(teacher, train_ds, caps, sigma_uni,
                                     device, seed=42 + int(eps * 10))
        cache_wf  = precompute_cache(teacher, train_ds, caps, sigma_wf,
                                     device, seed=42 + int(eps * 10))

        for mech, cache in [("uniform", cache_uni), ("channel_WF", cache_wf)]:
            print(f"\n  >>> ε={eps}  mechanism={mech}")
            dices = []
            for s in student_seeds:
                t0 = time.time()
                best, _ = train_student_distill(
                    train_ds, val_loader, cache, device,
                    student_base=16, teacher_base=32,
                    n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
                )
                dices.append(best)
                print(f"    seed={s}: best={best:.4f}  ({time.time()-t0:.1f}s)")
            mean_d = float(np.mean(dices))
            std_d  = float(np.std(dices))
            print(f"    → {mech}: {mean_d:.4f} ± {std_d:.4f}")
            results["sweep"][eps][mech] = {
                "dices": dices, "mean": mean_d, "std": std_d,
            }

    # Summary
    print("\n" + "=" * 110)
    print(f"clean teacher:                   {clean_dice:.4f}")
    print(f"student baseline ({n_seeds} seeds):       {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"{'ε':>5}  {'uniform Dice':>18}  {'WF Dice':>18}  {'paired lift (mean±SEM)':>26}  {'sig':>22}")
    print("-" * 110)
    for eps in epsilons:
        u_mean = results["sweep"][eps]["uniform"]["mean"]
        u_std  = results["sweep"][eps]["uniform"]["std"]
        w_mean = results["sweep"][eps]["channel_WF"]["mean"]
        w_std  = results["sweep"][eps]["channel_WF"]["std"]

        paired = [w - u for u, w in zip(
            results["sweep"][eps]["uniform"]["dices"],
            results["sweep"][eps]["channel_WF"]["dices"],
        )]
        lift_mean = float(np.mean(paired))
        lift_std  = float(np.std(paired))
        lift_sem  = lift_std / np.sqrt(len(paired))
        t_stat = lift_mean / lift_sem if lift_sem > 0 else 0
        wins   = sum(1 for p in paired if p > 0)

        if   abs(t_stat) >= 3: sig = f"*** {t_stat:+.1f}σ  ({wins}/{n_seeds} WF wins)"
        elif abs(t_stat) >= 2: sig = f"**  {t_stat:+.1f}σ  ({wins}/{n_seeds} WF wins)"
        elif abs(t_stat) >= 1: sig = f"~   {t_stat:+.1f}σ  ({wins}/{n_seeds} WF wins)"
        else:                  sig = f"≈   {t_stat:+.1f}σ  ({wins}/{n_seeds} WF wins)"

        results["sweep"][eps]["paired_lifts"] = paired
        results["sweep"][eps]["lift_mean"]    = lift_mean
        results["sweep"][eps]["lift_std"]     = lift_std
        results["sweep"][eps]["lift_sem"]     = lift_sem
        results["sweep"][eps]["t_stat"]       = t_stat
        results["sweep"][eps]["wins"]         = wins

        print(f"{eps:>5.1f}  {u_mean:.4f} ± {u_std:.4f}  "
              f"{w_mean:.4f} ± {w_std:.4f}  "
              f"{lift_mean:+.4f} ± {lift_sem:.4f}     {sig:>22}")
    print("=" * 110)

    out_path = Path(__file__).parent / "drive_student_distill_5seed_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel a
    ax = axes[0]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2,
               label=f"teacher clean = {clean_dice:.3f}")
    ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
               color="orange", alpha=0.18,
               label=f"student baseline {baseline_mean:.3f} ± {baseline_std:.3f}")
    ax.axhline(baseline_mean, color="orange", ls="--", lw=1.5)
    u_means = [results["sweep"][e]["uniform"]["mean"]    for e in epsilons]
    u_stds  = [results["sweep"][e]["uniform"]["std"]     for e in epsilons]
    w_means = [results["sweep"][e]["channel_WF"]["mean"] for e in epsilons]
    w_stds  = [results["sweep"][e]["channel_WF"]["std"]  for e in epsilons]
    ax.errorbar(epsilons, u_means, yerr=u_stds, fmt="s--", color="#d62728",
                lw=2, ms=9, capsize=6, label=f"uniform (mean ± std, {n_seeds}s)")
    ax.errorbar(epsilons, w_means, yerr=w_stds, fmt="o-", color="#1f77b4",
                lw=2, ms=9, capsize=6, label=f"channel-WF (mean ± std, {n_seeds}s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title(f"(a) Distilled-student Dice  ({n_seeds} seeds per cell)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel b
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    lift_means = [results["sweep"][e]["lift_mean"] for e in epsilons]
    lift_sems  = [results["sweep"][e]["lift_sem"]  for e in epsilons]
    for e in epsilons:
        for lift in results["sweep"][e]["paired_lifts"]:
            ax.scatter(e, lift, color="#2ca02c", alpha=0.4, s=45, zorder=3)
    ax.errorbar(epsilons, lift_means, yerr=lift_sems, fmt="D-", color="#2ca02c",
                lw=2.5, ms=12, capsize=6, label="mean ± SEM")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15,
                    label="±0.002 typical noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("paired lift:  channel-WF − uniform (per seed)")
    ax.set_title(f"(b) Paired WF − uniform lift (dots = {n_seeds} seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"DRIVE student distillation — {n_seeds}-seed paired comparison",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / "drive_student_distill_5seed.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
