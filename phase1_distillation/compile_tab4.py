"""
Compile paper Table 4 from existing 5-seed experiment results.

Tab. 4: vessel Dice on DRIVE for each mechanism, at the same user-level ε.
All cells are sample-once releases (matched threat model).

Mechanisms (all 5 seeds, paired student-init test where applicable):
  - K=1 uniform        — single-teacher baseline, uniform Gaussian
  - K=1 CANAL          — single-teacher, channel-WF allocation
  - PATE K=5 + uniform — multi-teacher aggregation, uniform on aggregate
  - PATE K=5 + CANAL   — multi-teacher aggregation, CANAL on aggregate
  - PATE K=10 + uniform — best K from saturation sweep

Data sources:
  K=1 uniform/CANAL, K=5 PATE+uniform:  drive_compare_pate_canal_uniform_results.json
  K=10 PATE+uniform:                    drive_pate_K_saturation_results.json
  K=5 PATE+CANAL:                       drive_pate_canal_combined_results.json
  Non-private reference:                drive_q1_per_iter_vs_sample_once_results.json
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load(name):
    return json.load(open(HERE / name))


def dice_block(d):
    """Return formatted 'mean ± std' string from a dict with 'dices' or 'mean'+'std'."""
    if "dices" in d:
        m, s = float(np.mean(d["dices"])), float(np.std(d["dices"]))
    else:
        m, s = d["mean"], d["std"]
    return f"{m:.4f} ± {s:.4f}", m, s


def paired_lift(a_dices, b_dices, label_a, label_b):
    paired = [a - b for a, b in zip(a_dices, b_dices)]
    pm = float(np.mean(paired))
    psd = float(np.std(paired))
    psem = psd / np.sqrt(len(paired))
    t = pm / psem if psem > 0 else 0
    return f"{pm:+.4f} ± {psem:.4f}  ({t:+.1f}σ)"


def main():
    comp  = load("drive_compare_pate_canal_uniform_results.json")
    ksat  = load("drive_pate_K_saturation_results.json")
    canal = load("drive_pate_canal_combined_results.json")
    q1    = load("drive_q1_per_iter_vs_sample_once_results.json")

    epsilons = [2.0, 8.0, 16.0]
    np_dice  = q1["non_private"]["mean"]

    print("=" * 110)
    print("Paper Table 4 — mechanism comparison on DRIVE (5 seeds per cell, sample-once framework)")
    print("=" * 110)
    print(f"  Non-private reference (ε=∞): vessel Dice = {np_dice:.4f}")
    print()
    print(f"{'mechanism':<28s}", end="")
    for e in epsilons:
        print(f"  ε={int(e):>3}                ", end="")
    print()
    print("-" * 110)

    rows = []

    # K=1 uniform
    label = "K=1 uniform (baseline)"
    cells = []
    for e in epsilons:
        d = comp["K1_uniform"][str(e)]
        s, m, sd = dice_block(d)
        cells.append((m, sd, d["dices"]))
    rows.append((label, cells))
    print(f"{label:<28s}", end="")
    for m, sd, _ in cells:
        print(f"  {m:.4f} ± {sd:.4f}    ", end="")
    print()

    # K=1 CANAL
    label = "K=1 CANAL (channel-WF)"
    cells = []
    for e in epsilons:
        d = comp["K1_CANAL"][str(e)]
        s, m, sd = dice_block(d)
        cells.append((m, sd, d["dices"]))
    rows.append((label, cells))
    print(f"{label:<28s}", end="")
    for m, sd, _ in cells:
        print(f"  {m:.4f} ± {sd:.4f}    ", end="")
    print()

    # K=5 PATE+uniform
    label = "PATE K=5 + uniform"
    cells = []
    for e in epsilons:
        d = comp["K5_PATE"][str(e)]
        s, m, sd = dice_block(d)
        cells.append((m, sd, d["dices"]))
    rows.append((label, cells))
    print(f"{label:<28s}", end="")
    for m, sd, _ in cells:
        print(f"  {m:.4f} ± {sd:.4f}    ", end="")
    print()

    # K=5 PATE+CANAL
    label = "PATE K=5 + CANAL"
    cells = []
    for e in epsilons:
        d = canal["sweep"][str(e)]["CANAL"]
        s, m, sd = dice_block(d)
        cells.append((m, sd, d["dices"]))
    rows.append((label, cells))
    print(f"{label:<28s}", end="")
    for m, sd, _ in cells:
        print(f"  {m:.4f} ± {sd:.4f}    ", end="")
    print()

    # K=10 PATE+uniform
    label = "PATE K=10 + uniform (best)"
    cells = []
    for e in epsilons:
        d = ksat["sweep"]["10"]["epsilons"][str(e)]
        s, m, sd = dice_block(d)
        cells.append((m, sd, d["dices"]))
    rows.append((label, cells))
    print(f"{label:<28s}", end="")
    for m, sd, _ in cells:
        print(f"  {m:.4f} ± {sd:.4f}    ", end="")
    print()

    print("=" * 110)

    # Paired tests vs K=1 uniform baseline
    print("\nPaired lift over K=1 uniform baseline  (5 seeds, paired student-init)")
    print("-" * 110)
    print(f"{'mechanism':<28s}", end="")
    for e in epsilons:
        print(f"  ε={int(e):>3}                  ", end="")
    print()
    print("-" * 110)
    base_dices_by_eps = [comp["K1_uniform"][str(e)]["dices"] for e in epsilons]

    for label, cells in rows[1:]:  # skip the uniform baseline itself
        print(f"{label:<28s}", end="")
        for (m, sd, dices), base_dices in zip(cells, base_dices_by_eps):
            paired = [d - b for d, b in zip(dices, base_dices)]
            pm = float(np.mean(paired))
            psem = float(np.std(paired) / np.sqrt(len(paired)))
            t = pm / psem if psem > 0 else 0
            print(f"  {pm:+.4f}±{psem:.4f} ({t:+.1f}σ)", end="")
        print()
    print("-" * 110)

    # Save tab4 as JSON for easy reuse
    out = {
        "non_private_reference": np_dice,
        "epsilons": epsilons,
        "rows": [],
    }
    for label, cells in rows:
        out["rows"].append({
            "label": label,
            "epsilons": {
                str(e): {"mean": m, "std": sd, "dices": dices}
                for e, (m, sd, dices) in zip(epsilons, cells)
            },
        })
    out_path = HERE / "tab4_compiled.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved compiled Tab. 4: {out_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(np_dice, color="#555", ls=":", lw=2.5,
               label=f"non-private (ε=∞) = {np_dice:.3f}")

    colors = ["#d62728", "#9467bd", "#1f77b4", "#8c564b", "#2ca02c"]
    markers = ["s", "^", "o", "v", "D"]

    for (label, cells), color, marker in zip(rows, colors, markers):
        means = [m for m, _, _ in cells]
        stds  = [sd for _, sd, _ in cells]
        ax.errorbar(epsilons, means, yerr=stds, fmt=f"{marker}-",
                    color=color, lw=2, ms=10, capsize=6, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(int(e)) for e in epsilons])
    ax.set_xlabel("privacy budget ε (user-level, sample-once)")
    ax.set_ylabel("vessel Dice")
    ax.set_title("Tab. 4 — mechanism comparison on DRIVE (5 seeds per cell)")
    ax.legend(loc="best", fontsize=9.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = HERE / "tab4_compiled.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved Tab. 4 plot: {plot_path}")


if __name__ == "__main__":
    main()
