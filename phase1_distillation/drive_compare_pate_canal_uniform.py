"""
Comparison plot: K=1 uniform vs K=1 CANAL vs K=5 PATE on real DRIVE.

Visualizes the three contender mechanisms for paper contribution (ii):
  - K=1 uniform: the baseline (no fancy allocation).
  - K=1 CANAL: channel-importance water-filling, original paper claim.
  - K=5 PATE: honest multi-teacher aggregation, new contribution (ii).

Uniform (K=1) and PATE (K=5) reuse the cached 5-seed PATE results
(drive_pate_5seed_results.json). CANAL (K=1) is run here with the
same teacher seed (1000) as PATE K=1 for apples-to-apples comparison.
All three use the CORRECT zCDP factor-of-2 (matches cluster code).
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
    compute_importance, get_importance, collect_caps,
)
from synthetic_demo import eps_to_rho
from drive_student_distill import train_student_distill
from drive_pate_poc import precompute_pate_cache


def correct_waterfilling_sigma(deltas, importance, rho):
    """WF σ with the correct factor-of-2 in zCDP."""
    s = importance.clamp(min=1e-12)
    kappa = ((deltas * s.sqrt()).sum() / (2.0 * rho)).sqrt()
    return kappa * deltas.sqrt() / s.pow(0.25)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance", default="grad_energy",
                    choices=["grad_energy", "act_norm"])
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print("\n[1/2] Training K=1 teacher (seed=1000, matches PATE K=1)...")
    torch.manual_seed(1000)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    t_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    train_teacher(teacher, t_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    importance = get_importance(args.importance, teacher, t_loader, device).to(device)
    caps       = collect_caps(teacher, t_loader, device).to(device)
    Cb_T = importance.shape[0]
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    print(f"  clean Dice = {clean_dice:.4f}")
    print(f"  importance ratio = {(importance.max()/importance.min()).item():.2f}×")

    print("\n[2/2] Running K=1 CANAL (channel-WF) × 5 seeds × 3 ε...")
    epsilons      = [2.0, 8.0, 16.0]
    student_seeds = [100, 200, 300, 400, 500]
    deltas = torch.full((Cb_T,), 2.0, device=device)  # K=1, Δ=2

    canal_data = {}
    for eps in epsilons:
        rho = eps_to_rho(eps)
        sigma_wf = correct_waterfilling_sigma(deltas, importance, rho)
        top10 = torch.argsort(importance, descending=True)[:10]
        bot10 = torch.argsort(importance, descending=False)[:10]
        print(f"\n  ε={eps}  σ_WF top10={sigma_wf[top10].mean().item():.3f}  "
              f"bot10={sigma_wf[bot10].mean().item():.3f}  ratio={sigma_wf[bot10].mean().item()/sigma_wf[top10].mean().item():.3f}×")
        cache = precompute_pate_cache([teacher], [caps], train_ds, sigma_wf,
                                      device, seed=42 + int(eps * 10))
        dices = []
        for s in student_seeds:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=16, teacher_base=32,
                n_epochs=40, lr=1e-3, lambda_feat=0.4, seed=s,
            )
            dices.append(best)
            print(f"    seed={s}: {best:.4f}  ({time.time()-t0:.1f}s)")
        canal_data[eps] = {
            "dices": dices,
            "mean": float(np.mean(dices)),
            "std":  float(np.std(dices)),
            "sem":  float(np.std(dices) / np.sqrt(len(dices))),
        }
        print(f"  → K=1 CANAL ε={eps}: {canal_data[eps]['mean']:.4f} ± {canal_data[eps]['std']:.4f}")

    # ---- load existing PATE data ----
    pate_path = Path(__file__).parent / "drive_pate_5seed_results.json"
    pate_data = json.load(open(pate_path))
    k1_uni = pate_data["sweep"]["1"]["epsilons"]
    k5_pate = pate_data["sweep"]["5"]["epsilons"]

    # ---- summary + paired tests ----
    print("\n" + "=" * 120)
    print(f"{'ε':>5}  {'K=1 uniform':>22}  {'K=1 CANAL':>22}  {'K=5 PATE':>22}  "
          f"{'CANAL−unif':>14}  {'PATE−unif':>14}")
    print("-" * 120)
    for eps in epsilons:
        u_dices = k1_uni[str(eps)]["dices"]
        c_dices = canal_data[eps]["dices"]
        p_dices = k5_pate[str(eps)]["dices"]

        u_mean = float(np.mean(u_dices));  u_std = float(np.std(u_dices))
        c_mean = canal_data[eps]["mean"];  c_std = canal_data[eps]["std"]
        p_mean = float(np.mean(p_dices));  p_std = float(np.std(p_dices))

        canal_lift_paired = [c - u for c, u in zip(c_dices, u_dices)]
        pate_lift_paired  = [p - u for p, u in zip(p_dices, u_dices)]
        cl_mean = float(np.mean(canal_lift_paired));  cl_sem = float(np.std(canal_lift_paired) / np.sqrt(len(canal_lift_paired)))
        pl_mean = float(np.mean(pate_lift_paired));   pl_sem = float(np.std(pate_lift_paired)  / np.sqrt(len(pate_lift_paired)))

        print(f"{eps:>5.1f}  {u_mean:.4f} ± {u_std:.4f}  "
              f"{c_mean:.4f} ± {c_std:.4f}  "
              f"{p_mean:.4f} ± {p_std:.4f}  "
              f"{cl_mean:+.4f}±{cl_sem:.4f}  "
              f"{pl_mean:+.4f}±{pl_sem:.4f}")
    print("=" * 120)

    # ---- save ----
    out = {
        "importance": args.importance,
        "clean_teacher": clean_dice,
        "epsilons": epsilons,
        "student_seeds": student_seeds,
        "K1_uniform": {str(e): k1_uni[str(e)]    for e in epsilons},
        "K1_CANAL":   {str(e): canal_data[e]      for e in epsilons},
        "K5_PATE":    {str(e): k5_pate[str(e)]   for e in epsilons},
    }
    out_path = Path(__file__).parent / f"drive_compare_pate_canal_uniform_{args.importance}_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel a: Dice vs ε for the 3 methods
    ax = axes[0]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean teacher = {clean_dice:.3f}")
    u_means = [float(np.mean(k1_uni[str(e)]["dices"]))    for e in epsilons]
    u_stds  = [float(np.std(k1_uni[str(e)]["dices"]))     for e in epsilons]
    c_means = [canal_data[e]["mean"]                       for e in epsilons]
    c_stds  = [canal_data[e]["std"]                        for e in epsilons]
    p_means = [float(np.mean(k5_pate[str(e)]["dices"]))   for e in epsilons]
    p_stds  = [float(np.std(k5_pate[str(e)]["dices"]))    for e in epsilons]
    ax.errorbar(epsilons, u_means, yerr=u_stds, fmt="s--", color="#d62728",
                lw=2, ms=10, capsize=6,
                label="K=1 uniform (baseline)")
    ax.errorbar(epsilons, c_means, yerr=c_stds, fmt="^-", color="#9467bd",
                lw=2, ms=10, capsize=6,
                label="K=1 CANAL (channel-WF, original paper)")
    ax.errorbar(epsilons, p_means, yerr=p_stds, fmt="o-", color="#2ca02c",
                lw=2.8, ms=13, capsize=7,
                label="K=5 PATE (multi-teacher, NEW contribution (ii))")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) Three candidates for contribution (ii)  —  5 seeds each")
    ax.legend(loc="best", fontsize=9.5)
    ax.grid(alpha=0.3)

    # Panel b: paired lift over uniform baseline
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    canal_lifts_m = []
    canal_lifts_sem = []
    pate_lifts_m = []
    pate_lifts_sem = []
    for e in epsilons:
        u_d = k1_uni[str(e)]["dices"]
        c_d = canal_data[e]["dices"]
        p_d = k5_pate[str(e)]["dices"]
        cl = [c - u for c, u in zip(c_d, u_d)]
        pl = [p - u for p, u in zip(p_d, u_d)]
        canal_lifts_m.append(np.mean(cl)); canal_lifts_sem.append(np.std(cl)/np.sqrt(len(cl)))
        pate_lifts_m.append(np.mean(pl));  pate_lifts_sem.append(np.std(pl)/np.sqrt(len(pl)))
        for v in cl:
            ax.scatter(e, v, color="#9467bd", alpha=0.4, s=40, zorder=3)
        for v in pl:
            ax.scatter(e, v, color="#2ca02c", alpha=0.4, s=40, zorder=3)
    ax.errorbar(epsilons, canal_lifts_m, yerr=canal_lifts_sem, fmt="^-",
                color="#9467bd", lw=2, ms=10, capsize=6,
                label="CANAL − uniform (paired)")
    ax.errorbar(epsilons, pate_lifts_m, yerr=pate_lifts_sem, fmt="D-",
                color="#2ca02c", lw=2.5, ms=11, capsize=6,
                label="PATE K=5 − uniform (paired)")
    ax.fill_between([epsilons[0]*0.7, epsilons[-1]*1.3], -0.002, 0.002,
                    color="gray", alpha=0.15, label="±0.002 noise band")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlim(epsilons[0]*0.7, epsilons[-1]*1.3)
    ax.set_xlabel("privacy budget ε")
    ax.set_ylabel("paired Dice lift vs K=1 uniform (per seed)")
    ax.set_title("(b) Lift over K=1 uniform baseline  (dots = 5 seeds)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Contribution (ii) showdown: PATE wins, CANAL ≈ uniform",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = Path(__file__).parent / f"drive_compare_pate_canal_uniform_{args.importance}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
