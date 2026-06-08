"""
Sensitivity sweep: simulate Δ_c = 2/K for K ∈ {1, 2, 5, 10, 50, 100, 1000},
run the noise probe at each, see if WF beats uniform when σ shrinks.

CAVEAT: This is utility simulation, NOT honest DP. To legitimately get
Δ = 2/K we'd need a real K-teacher PATE ensemble (direction A from prior
discussion). Here we just plug Δ = 2/K into the σ formula and measure Dice.

The question we're testing: would lowering sensitivity (by any means) make
the channel-WF vs uniform gap meaningful? If yes → PATE is worth pursuing.
If still ≈ 0 across all K → the ¼-power compression of importance is the
real bottleneck and no Δ trick will fix it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import (
    DriveDataset, TinyUNet, train_teacher, evaluate_vessel_dice,
    compute_importance, collect_caps, vessel_dice,
)
from synthetic_demo import (
    eps_to_rho, waterfilling_sigma, uniform_sigma,
    clip_and_normalise, denormalise,
)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    print("\n[Setup] Loading DRIVE + training teacher...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    importance = compute_importance(teacher, train_loader, device).to(device)
    caps       = collect_caps(teacher, train_loader, device).to(device)
    Cb = importance.shape[0]
    print(f"  clean Dice = {clean_dice:.4f}")
    print(f"  importance ratio = {(importance.max()/importance.min()).item():.2f}×")

    K_values  = [1, 2, 5, 10, 50, 100, 1000]
    epsilons  = [1.0, 2.0, 4.0, 8.0, 16.0]

    results = {"clean": clean_dice, "K_values": K_values, "epsilons": epsilons,
               "sweep": {}}

    print("\n" + "=" * 96)
    print(f"{'K':>5}  {'Δ':>9}  {'ε':>5}  {'σ_uni':>8}  {'σ_WF top':>10}  "
          f"{'σ_WF bot':>10}  {'uniform':>9}  {'WF':>9}  {'lift':>9}")
    print("=" * 96)

    teacher.eval()
    for K in K_values:
        deltas = torch.full((Cb,), 2.0 / K, device=device)
        results["sweep"][K] = {}

        for eps in epsilons:
            rho = eps_to_rho(eps)
            sigma_uni = uniform_sigma(deltas, rho)
            sigma_wf  = waterfilling_sigma(deltas, importance, rho)

            top10 = torch.argsort(importance, descending=True)[:10]
            bot10 = torch.argsort(importance, descending=False)[:10]

            uni_dices, wf_dices = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    e1, e2, e3 = teacher.encode(x)
                    bn_norm = clip_and_normalise(e3, caps)
                    B, C, H, W = bn_norm.shape

                    # SAME random seed → identical underlying N(0,1) draws.
                    # Only σ differs between uniform and WF → fair comparison.
                    torch.manual_seed(42 + int(eps * 100) + K)
                    raw = torch.randn(B, C, H, W, device=device)

                    bn_uni = denormalise(bn_norm + raw * sigma_uni.view(1, C, 1, 1), caps)
                    uni_dices.append(vessel_dice(teacher.decode(e1, e2, bn_uni), y))

                    bn_wf  = denormalise(bn_norm + raw * sigma_wf.view(1, C, 1, 1),  caps)
                    wf_dices.append(vessel_dice(teacher.decode(e1, e2, bn_wf), y))

            uni_mean = float(np.mean(uni_dices))
            wf_mean  = float(np.mean(wf_dices))
            lift     = wf_mean - uni_mean

            print(f"{K:>5}  {2.0/K:>9.4f}  {eps:>5.1f}  {sigma_uni[0].item():>8.4f}  "
                  f"{sigma_wf[top10].mean().item():>10.4f}  "
                  f"{sigma_wf[bot10].mean().item():>10.4f}  "
                  f"{uni_mean:>9.4f}  {wf_mean:>9.4f}  {lift:>+9.4f}")

            results["sweep"][K][eps] = {
                "sigma_uniform":    float(sigma_uni[0]),
                "sigma_wf_top10":   float(sigma_wf[top10].mean()),
                "sigma_wf_bot10":   float(sigma_wf[bot10].mean()),
                "uniform_dice":     uni_mean,
                "wf_dice":          wf_mean,
                "lift":             lift,
            }
        print("-" * 96)
    print("=" * 96)

    out_path = Path(__file__).parent / "drive_sensitivity_sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved JSON: {out_path}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel a: Dice for uniform & WF, lines per K, x-axis = ε
    ax = axes[0]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean = {clean_dice:.3f}")
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(K_values)))
    for K, color in zip(K_values, colors):
        uni = [results["sweep"][K][e]["uniform_dice"] for e in epsilons]
        wf  = [results["sweep"][K][e]["wf_dice"]      for e in epsilons]
        ax.plot(epsilons, uni, "s--", color=color, lw=1.5, ms=6, alpha=0.55)
        ax.plot(epsilons, wf,  "o-",  color=color, lw=2,   ms=7,
                label=f"K={K}  (Δ={2/K:.3g})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget  ε")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) Dice across (ε, K): solid = WF, dashed = uniform")
    ax.legend(loc="lower right", fontsize=9, ncol=1)
    ax.grid(alpha=0.3)

    # Panel b: WF lift heatmap
    ax = axes[1]
    lift_mat = np.zeros((len(K_values), len(epsilons)))
    for i, K in enumerate(K_values):
        for j, e in enumerate(epsilons):
            lift_mat[i, j] = results["sweep"][K][e]["lift"]
    vmax = max(abs(lift_mat.min()), abs(lift_mat.max()), 0.001)
    im = ax.imshow(lift_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"ε={int(e)}" for e in epsilons])
    ax.set_yticks(range(len(K_values)))
    ax.set_yticklabels([f"K={K}\nΔ={2/K:.3g}" for K in K_values])
    ax.set_xlabel("privacy budget")
    ax.set_ylabel("sensitivity scale (K = simulated PATE size)")
    ax.set_title("(b) WF − uniform lift (Dice)")
    plt.colorbar(im, ax=ax, label="Dice lift")
    for i in range(len(K_values)):
        for j in range(len(epsilons)):
            v = lift_mat[i, j]
            color = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.suptitle(
        f"DRIVE — does shrinking Δ widen the WF vs uniform gap?  "
        f"(C={Cb}, importance ratio≈{(importance.max()/importance.min()).item():.1f}×)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(__file__).parent / "drive_sensitivity_sweep.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
