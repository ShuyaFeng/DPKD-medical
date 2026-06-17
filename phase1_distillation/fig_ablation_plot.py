"""Honest pruning ablation figure: importance vs random selection, vs floor/ceiling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.load(open(HERE / "drive_pruning_ablation_results.json"))
EPS = [2.0, 8.0, 16.0]
def vd(cell): return cell["vessel_dice"]["mean"], cell["vessel_dice"].get("std", 0)/np.sqrt(len(d["seeds"]))

floor   = d["no_noise"]["student_only"]["vessel_dice"]["mean"]   # GT only, no teacher
ceiling = d["no_noise"]["clean_full"]["vessel_dice"]["mean"]     # clean teacher distill

conds = [
    ("noisy_full (128 ch)",  "noisy_full",  "#7f7f7f"),
    ("imp keep-10%",         "imp_keep10",  "#1f77b4"),
    ("rand keep-10%",        "rand_keep10", "#aec7e8"),
    ("imp keep-2%",          "imp_keep02",  "#d62728"),
    ("rand keep-2%",         "rand_keep02", "#ff9896"),
]
fig, ax = plt.subplots(figsize=(11, 6.2))
nb = len(conds); w = 0.15; x = np.arange(len(EPS))
for i,(name,key,col) in enumerate(conds):
    ms = [vd(d["noisy"][str(e)][key])[0] for e in EPS]
    ss = [vd(d["noisy"][str(e)][key])[1] for e in EPS]
    ax.bar(x + (i-nb/2+0.5)*w, ms, w, yerr=ss, capsize=3, color=col, label=name)
ax.axhline(floor,   color="black", ls="--", lw=1.6, label=f"student-only FLOOR (no teacher) {floor:.3f}")
ax.axhline(ceiling, color="green", ls=":",  lw=1.6, label=f"clean-teacher CEILING {ceiling:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS])
ax.set_ylabel("vessel Dice (5 seeds, mean ± SEM)")
ax.set_ylim(0.55, 0.69)
ax.set_title("Honest pruning ablation: importance ≈ random (gain is dimensionality, not selection)\n"
             "single teacher K=1 — and at ε=2 noisy distillation barely clears the no-teacher floor")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE/"fig_ablation.png", dpi=140)
print("saved fig_ablation.png")

# paired imp-vs-rand deltas
print("\n=== imp − rand paired (vessel Dice) ===")
for e in EPS:
    for keep,ik,rk in [("10%","imp_keep10","rand_keep10"),("2%","imp_keep02","rand_keep02")]:
        iv=np.array(d["noisy"][str(e)][ik]["vessel_dice"]["vals"])
        rv=np.array(d["noisy"][str(e)][rk]["vessel_dice"]["vals"])
        diff=iv-rv
        print(f"  ε={int(e):>2} keep {keep}: imp−rand = {diff.mean():+.4f} ± {diff.std()/np.sqrt(len(diff)):.4f}")
print(f"\nfloor(no teacher)={floor:.4f}  ceiling(clean)={ceiling:.4f}")
print("noisy methods at/below floor (teacher useless):")
for e in EPS:
    below=[k for k in d["noisy"][str(e)] if d["noisy"][str(e)][k]["vessel_dice"]["mean"] <= floor]
    print(f"  ε={int(e):>2}: {below}")
