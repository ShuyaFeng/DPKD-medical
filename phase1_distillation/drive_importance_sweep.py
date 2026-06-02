"""
Importance-ratio ablation: does channel-WF beat uniform if the importance
spectrum were artificially steeper?

We reuse the trained DRIVE teacher from drive_local_demo.py, then for each
ratio R ∈ {1, 10, 100, 1000, 10000, 100000} we *synthesize* an importance
vector that:
  - preserves the original ranking from the teacher's bottleneck gradients
  - geometrically interpolates magnitudes from 1 (least) to R (most)

So R=1.8 reproduces our observed natural spectrum. R=1 collapses WF to
uniform. R=10⁵ is the most extreme skew we can realistically test.

For each R we run the same noise probe (clean / uniform / channel_WF) at
ε ∈ {4, 8, 16} and compare. Output: numbers + plot.

The key question: at what R does WF actually pull away from uniform — and
when it does, does it approach clean, or just get less bad?
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
    compute_importance, collect_caps, noise_probe,
)


# --------------------------------------------------------------------------
# Synthesise importance with controlled ratio
# --------------------------------------------------------------------------

def synthesize_importance(original: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Preserve the original importance ranking; replace magnitudes with a
    geometric sweep from 1 (least important) to `ratio` (most important).
    """
    if ratio <= 1.0:
        return torch.ones_like(original)
    C = original.shape[0]
    rank_order = torch.argsort(original, descending=True)        # most-important first
    new_imp = torch.empty_like(original)
    for i, ch in enumerate(rank_order.tolist()):
        # i=0       → most important → value = ratio
        # i=C-1     → least           → value = 1
        new_imp[ch] = ratio ** ((C - 1 - i) / (C - 1))
    return new_imp


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    print(f"Device: {device}")

    # ----------------------------------------------------------------------
    # Load data + train teacher (reuse drive_local_demo's pieces)
    # ----------------------------------------------------------------------
    print("\n[1/4] Loading DRIVE + training teacher (same as drive_local_demo)...")
    train_ds = DriveDataset("train", size=96)
    val_ds   = DriveDataset("val",   size=96)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    teacher = TinyUNet(in_ch=3, num_classes=2, base=32).to(device)
    train_teacher(teacher, train_loader, n_epochs=60, lr=1e-3, device=device)
    clean_dice = evaluate_vessel_dice(teacher, val_loader, device)
    print(f"  clean teacher vessel Dice = {clean_dice:.4f}")

    # ----------------------------------------------------------------------
    # Importance + caps
    # ----------------------------------------------------------------------
    print("\n[2/4] Get natural importance + caps...")
    natural = compute_importance(teacher, train_loader, device)
    caps    = collect_caps(teacher, train_loader, device)
    nat_ratio = float(natural.max() / natural.min())
    print(f"  natural importance ratio = {nat_ratio:.2f}×  (this is what 'real' DRIVE gives us)")

    # ----------------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------------
    print("\n[3/4] Sweeping synthetic importance ratios...")
    ratios = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    epsilons = [4.0, 8.0, 16.0]
    K = 10

    table = {}
    for R in ratios:
        synth = synthesize_importance(natural, R).to(device)
        actual_ratio = float(synth.max() / synth.min())

        torch.manual_seed(42)                            # same noise draw across R
        means, sigma_table = noise_probe(
            teacher, val_loader, synth, caps, device, epsilons, K=K,
        )
        for e in epsilons:
            sw = sigma_table[e]["channel_WF"]
            top10 = torch.argsort(synth, descending=True)[:10]
            bot10 = torch.argsort(synth, descending=False)[:10]
            sig_ratio = float(sw[bot10].mean() / sw[top10].mean())

            u = means["uniform"][e]
            w = means["channel_WF"][e]
            table[(R, e)] = {
                "uniform":   u,
                "wf":        w,
                "lift":      w - u,
                "sig_ratio": sig_ratio,
                "imp_ratio": actual_ratio,
            }

        # Pretty print
        print(f"  R={R:>8.0f} (imp ratio={actual_ratio:>10.2f}×):")
        for e in epsilons:
            r = table[(R, e)]
            print(f"      ε={e:>4.1f}  uniform={r['uniform']:.4f}  "
                  f"WF={r['wf']:.4f}  lift={r['lift']:+.4f}  "
                  f"σ_ratio={r['sig_ratio']:.2f}×")

    # ----------------------------------------------------------------------
    # Save + plot
    # ----------------------------------------------------------------------
    print("\n[4/4] Saving + plotting...")

    out = {
        "natural_importance_ratio": nat_ratio,
        "clean_dice": clean_dice,
        "K": K,
        "sweep": [
            {"importance_ratio": R, "epsilon": e, **vals}
            for (R, e), vals in table.items()
        ],
    }
    out_path = Path(__file__).parent / "drive_importance_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  saved: {out_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"DRIVE — would channel-WF beat uniform if importance ratio were higher? "
        f"(K={K}, natural ratio {nat_ratio:.1f}×)",
        fontsize=13, fontweight="bold",
    )

    colors = {4.0: "#9467bd", 8.0: "#1f77b4", 16.0: "#2ca02c"}

    # Panel a: Dice vs importance ratio, all 3 mechanisms
    ax = axes[0]
    ax.axhline(clean_dice, color="#555", ls=":", lw=2.5,
               label=f"clean (no noise) = {clean_dice:.3f}")
    for e in epsilons:
        uni = [table[(R, e)]["uniform"] for R in ratios]
        wf  = [table[(R, e)]["wf"]      for R in ratios]
        ax.plot(ratios, uni, "s--", color=colors[e], lw=1.5, ms=6, alpha=0.5,
                label=f"uniform ε={e:.0f}")
        ax.plot(ratios, wf,  "o-",  color=colors[e], lw=2,   ms=8,
                label=f"channel-WF ε={e:.0f}")
    ax.axvline(nat_ratio, color="orange", ls="-.", lw=1.5,
               label=f"natural ratio ({nat_ratio:.1f}×)")
    ax.set_xscale("log")
    ax.set_xlabel("synthetic importance ratio  s_max / s_min")
    ax.set_ylabel("vessel Dice")
    ax.set_title("(a) Does WF approach clean as importance gets skewed?")
    ax.legend(fontsize=8, loc="center right", ncol=2)
    ax.grid(alpha=0.3)

    # Panel b: WF - uniform lift vs importance ratio
    ax = axes[1]
    ax.axhline(0, color="black", lw=1)
    for e in epsilons:
        lifts = [table[(R, e)]["lift"] for R in ratios]
        ax.plot(ratios, lifts, "o-", color=colors[e], lw=2, ms=8,
                label=f"ε={e:.0f}")
    ax.axvline(nat_ratio, color="orange", ls="-.", lw=1.5,
               label=f"natural ratio ({nat_ratio:.1f}×)")
    ax.set_xscale("log")
    ax.set_xlabel("synthetic importance ratio  s_max / s_min")
    ax.set_ylabel("Dice lift:  WF − uniform")
    ax.set_title("(b) Where the WF advantage actually shows up")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = Path(__file__).parent / "drive_importance_sweep.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  saved: {plot_path}")


if __name__ == "__main__":
    main()
