"""Plot Phase 1 Dice across mechanisms and noise multipliers."""

import json
import sys
import statistics
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aggregate(files):
    all_r = []
    for f in files:
        with open(f) as fh:
            all_r.extend(json.load(fh))

    agg = defaultdict(list)
    for r in all_r:
        key = (r["mechanism"], r.get("noise_multiplier", 0.0))
        agg[key].append(r["best_val_dice"])

    mech_set = sorted({k[0] for k in agg.keys()})
    nm_set = sorted({k[1] for k in agg.keys() if k[0] != "no-dp"})

    table = {}
    for m in mech_set:
        for nm in (nm_set if m != "no-dp" else [0.0]):
            vs = agg.get((m, nm), [])
            if vs:
                table[(m, nm)] = (
                    statistics.mean(vs),
                    statistics.stdev(vs) if len(vs) > 1 else 0.0,
                    len(vs),
                )
    return table, mech_set, nm_set


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else ["phase1_results_sweep.json"]
    table, mech_set, nm_set = aggregate(files)

    # Pretty print
    print(f"{'mechanism':<14} " + " ".join(f"nm={nm:<8.3f}" for nm in nm_set))
    print("-" * (14 + 12 * len(nm_set)))
    order = ["no-dp", "uniform", "channel-WF", "spatial-WF", "joint-WF", "joint-WF+thr"]
    for m in order:
        if m not in mech_set:
            continue
        if m == "no-dp":
            v, s, n = table.get((m, 0.0), (None, None, 0))
            if v is not None:
                print(f"{m:<14} (no-DP ref) {v:.3f} ± {s:.3f} (n={n})")
            continue
        cells = []
        for nm in nm_set:
            v, s, n = table.get((m, nm), (None, None, 0))
            if v is None:
                cells.append("   -      ")
            else:
                cells.append(f"{v:.3f}±{s:.2f}")
        print(f"{m:<14} " + " ".join(f"{c:<11}" for c in cells))

    # Plot: x = noise multiplier (log), y = Dice, one curve per mechanism
    colors = {
        "no-dp": "#2ca02c",
        "uniform": "gray",
        "channel-WF": "tab:blue",
        "spatial-WF": "tab:green",
        "joint-WF": "tab:red",
        "joint-WF+thr": "tab:orange",
    }
    markers = {
        "no-dp": "*", "uniform": "o", "channel-WF": "s",
        "spatial-WF": "^", "joint-WF": "D", "joint-WF+thr": "P",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in order:
        if m == "no-dp" or m not in mech_set:
            continue
        means = [table.get((m, nm), (np.nan, 0, 0))[0] for nm in nm_set]
        stds = [table.get((m, nm), (0, 0, 0))[1] for nm in nm_set]
        means = np.array(means); stds = np.array(stds)
        lw = 2.5 if m in ("joint-WF", "joint-WF+thr") else 1.8
        ax.plot(nm_set, means, label=m, color=colors[m], marker=markers[m],
                markersize=8, linewidth=lw)
        ax.fill_between(nm_set, means - stds, means + stds,
                        color=colors[m], alpha=0.15)

    # No-DP horizontal line
    if "no-dp" in mech_set:
        nodp = table[("no-dp", 0.0)][0]
        ax.axhline(nodp, color=colors["no-dp"], linestyle=":", linewidth=2,
                   label=f"no-DP ({nodp:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("noise multiplier (smaller = more privacy if Δ kept fixed)")
    ax.set_ylabel("best validation Dice (DRIVE, 96×96)")
    ax.set_title("Phase 1: U-Net vessel segmentation Dice under DP image release\n"
                 r"data-independent prior (avg of 10 train masks), 3 seeds × 5 mechs × 4 noise levels")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = "phase1_dice.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
