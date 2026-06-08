"""
Multi-seed student distillation: WF vs uniform with 3 student seeds.

Same setup as drive_student_distill.py, but each (ε, mechanism) cell is
run with 3 fresh student seeds. Computes:
  - mean ± std Dice per (ε, mech)
  - paired WF - uniform lift per seed
  - lift mean ± SEM   → statistical-significance signal
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

    # -- teacher --
    print("\n[1/3] Training teacher...")
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

    # -- 3 student baselines --
    print("\n[2/3] Student baselines (3 seeds, no distillation)...")
    baseline_dices = []
    for s in [100, 200, 300]:
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

    # -- distillation sweep --
    print("\n[3/3] Student distillation: WF vs uniform × 3 seeds each...")
    epsilons = [2.0, 8.0, 16.0]
    K = 10
    student_seeds = [100, 200, 300]
    deltas = torch.full((Cb_T,), 2.0 / K, device=device)

    results = {
        "clean_teacher":         clean_dice,
        "student_baseline_mean": baseline_mean,
        "student_baseline_std":  baseline_std,
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

        # Precompute (same noise per (ε, mech) across student seeds)
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

    # -- summary --
    print("\n" + "=" * 100)
    print(f"clean teacher:               {clean_dice:.4f}")
    print(f"student baseline (3 seeds):  {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"{'ε':>5}  {'uniform Dice':>18}  {'WF Dice':>18}  {'paired lift':>14}  {'sig':>20}")
    print("-" * 100)
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

        if   lift_mean >  2 * lift_sem: sig = "** WF wins (>2σ)"
        elif lift_mean < -2 * lift_sem: sig = "** uniform wins (>2σ)"
        elif abs(lift_mean) > lift_sem: sig = "~ trend"
        else:                            sig = "≈ tied (noise)"

        results["sweep"][eps]["paired_lifts"]  = paired
        results["sweep"][eps]["lift_mean"]     = lift_mean
        results["sweep"][eps]["lift_std"]      = lift_std
        results["sweep"][eps]["lift_sem"]      = lift_sem

        print(f"{eps:>5.1f}  {u_mean:.4f} ± {u_std:.4f}  "
              f"{w_mean:.4f} ± {w_std:.4f}  "
              f"{lift_mean:+.4f}±{lift_sem:.4f}  {sig:>20}")
    print("=" * 100)

    out_path = Path(__file__).parent / "drive_student_distill_multiseed_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # -- plot --
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
                lw=2, ms=9, capsize=6, label="uniform (mean ± std)")
    ax.errorbar(epsilons, w_means, yerr=w_stds, fmt="o-", color="#1f77b4",
                lw=2, ms=9, capsize=6, label="channel-WF (mean ± std)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) Distilled-student Dice  (3 seeds per cell)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel b
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    lift_means = [results["sweep"][e]["lift_mean"] for e in epsilons]
    lift_sems  = [results["sweep"][e]["lift_sem"]  for e in epsilons]
    for e in epsilons:
        for lift in results["sweep"][e]["paired_lifts"]:
            ax.scatter(e, lift, color="#2ca02c", alpha=0.4, s=40, zorder=3)
    ax.errorbar(epsilons, lift_means, yerr=lift_sems, fmt="D-", color="#2ca02c",
                lw=2.5, ms=11, capsize=6, label="mean ± SEM")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15,
                    label="±0.002 typical noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("paired lift:  channel-WF − uniform (per seed)")
    ax.set_title("(b) Paired WF − uniform lift  (dots = seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("DRIVE student distillation — multi-seed comparison",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / "drive_student_distill_multiseed.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
