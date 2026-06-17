"""
Analysis + figures for the channel-axis story:
  (1) CANAL (water-filling allocation) is empirically dead  -> fig_canal_ceiling.png
  (2) Channel PRUNING (importance drop) is the real winner   -> fig_pruning_sweep.png
  (3) Unified mechanism comparison at matched epsilon         -> fig_comparison.png

All Dice numbers are on the SAME TinyUNet student protocol (5 seeds), so the
files below are mutually comparable (K1_uniform matches across files).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
def load(name): return json.load(open(HERE / name))

prune   = load("drive_single_teacher_pruning_results.json")
compare = load("drive_compare_pate_canal_uniform_results.json")
combined= load("drive_pate_canal_combined_results.json")
impsw   = load("drive_importance_sweep_results.json")
ksat    = load("drive_pate_K_saturation_results.json")

EPS = [2.0, 8.0, 16.0]
clean_ref = compare["clean_teacher"]            # ~0.624 single clean teacher, this protocol

# =====================================================================
# FIG 1 — CANAL ceiling: paired lift ~ 0, and synthetic R-sweep flat
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) CANAL - uniform paired lift at K=1 and K=5
ax = axes[0]
# K=1 from compare file
k1_lift = [compare["K1_CANAL"][str(e)]["mean"] - compare["K1_uniform"][str(e)]["mean"] for e in EPS]
# K=5 from combined file (has paired_lift_mean/sem)
k5_lift = [combined["sweep"][str(e)]["paired_lift_mean"] for e in EPS]
k5_sem  = [combined["sweep"][str(e)]["paired_lift_sem"]  for e in EPS]
ax.axhline(0, color="black", lw=1)
ax.fill_between([1.5, 20], -0.002, 0.002, color="gray", alpha=0.15, label="±0.002 seed-noise band")
ax.plot(EPS, k1_lift, "o-", color="#1f77b4", lw=2, ms=10, label="CANAL − uniform, K=1")
ax.errorbar(EPS, k5_lift, yerr=k5_sem, fmt="s-", color="#d62728", lw=2, ms=9, capsize=5, label="CANAL − uniform, K=5 (paired)")
ax.set_xscale("log", base=2); ax.set_xticks(EPS); ax.set_xticklabels([str(int(e)) for e in EPS])
ax.set_xlim(1.6, 20)
ax.set_xlabel("privacy budget ε"); ax.set_ylabel("Dice lift over uniform")
ax.set_title("(a) CANAL allocation lift ≈ 0 (within seed noise)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) synthetic importance-ratio sweep: lift vs R at eps=16
ax = axes[1]
rows = impsw["sweep"]
Rs = sorted({r["importance_ratio"] for r in rows})
for e, col in zip([4.0, 8.0, 16.0], ["#2ca02c", "#ff7f0e", "#9467bd"]):
    lifts = [next(r["lift"] for r in rows if r["importance_ratio"]==R and r["epsilon"]==e) for R in Rs]
    ax.plot(Rs, lifts, "o-", color=col, lw=2, ms=8, label=f"ε={int(e)}")
ax.axhline(0, color="black", lw=1)
ax.axvline(impsw["natural_importance_ratio"], color="red", ls="--", lw=1.5,
           label=f"natural R={impsw['natural_importance_ratio']:.2f}")
ax.fill_between([1, max(Rs)], -0.002, 0.002, color="gray", alpha=0.15)
ax.set_xscale("log"); ax.set_xlabel("synthetic importance ratio R = max s / min s")
ax.set_ylabel("WF − uniform lift"); ax.set_title("(b) Even at R=1000× the lift stays flat (R^{1/4} ceiling)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle("CANAL (channel water-filling allocation): no empirical gain on DRIVE", fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(HERE/"fig_canal_ceiling.png", dpi=140)
print("saved fig_canal_ceiling.png")

# =====================================================================
# FIG 2 — Pruning sweep: Dice vs keep-fraction, per eps
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 6))
kf = prune["keep_fractions"]                       # [1.0,0.5,0.25,0.1,0.05,0.02]
nact = {f: prune["sweep"][str(EPS[0])][f"{f:.2f}"]["n_active"] for f in kf}
xs = list(range(len(kf)))                          # categorical, most->least kept
colors = {2.0:"#1f77b4", 8.0:"#ff7f0e", 16.0:"#2ca02c"}
for e in EPS:
    means = [prune["sweep"][str(e)][f"{f:.2f}"]["mean"] for f in kf]
    sems  = [prune["sweep"][str(e)][f"{f:.2f}"]["sem"]  for f in kf]
    ax.errorbar(xs, means, yerr=sems, fmt="o-", color=colors[e], lw=2, ms=9, capsize=5, label=f"prune, ε={int(e)}")
    # PATE K=10 reference dashed line at this eps
    ax.axhline(prune["pate_k10_reference"][str(e)], color=colors[e], ls="--", lw=1.3, alpha=0.7)
ax.axhline(clean_ref, color="black", ls=":", lw=1.5, label=f"clean teacher ({clean_ref:.3f})")
ax.set_xticks(xs)
ax.set_xticklabels([f"{int(f*100)}%\n({nact[f]} ch)" for f in kf])
ax.set_xlabel("channels kept  (most important →)   |   dashed = PATE K=10 same-ε reference")
ax.set_ylabel("vessel Dice (5 seeds, mean ± SEM)")
ax.set_title("Channel pruning: aggressive drop (keep 2–5%) beats keeping all — and beats PATE K=10")
ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(HERE/"fig_pruning_sweep.png", dpi=140)
print("saved fig_pruning_sweep.png")

# =====================================================================
# FIG 3 — Unified mechanism comparison (the money figure)
# =====================================================================
def prune_at(frac, e):
    c = prune["sweep"][str(e)][f"{frac:.2f}"]; return c["mean"], c["sem"]

methods = [
    ("K=1 uniform (baseline)",  lambda e:(compare["K1_uniform"][str(e)]["mean"], compare["K1_uniform"][str(e)]["sem"]), "#7f7f7f"),
    ("K=1 + CANAL (alloc)",     lambda e:(compare["K1_CANAL"][str(e)]["mean"],   compare["K1_CANAL"][str(e)]["sem"]),   "#1f77b4"),
    ("PATE K=10 (sensitivity)", lambda e:(ksat["sweep"]["10"]["epsilons"][f"{e}"]["mean"], ksat["sweep"]["10"]["epsilons"][f"{e}"]["sem"]), "#ff7f0e"),
    ("K=1 prune keep-5%",       lambda e:prune_at(0.05, e), "#2ca02c"),
    ("K=1 prune keep-2% (BEST)",lambda e:prune_at(0.02, e), "#d62728"),
]
fig, ax = plt.subplots(figsize=(11, 6))
nb = len(methods); w = 0.15
x = np.arange(len(EPS))
for i,(name,fn,col) in enumerate(methods):
    ms=[fn(e)[0] for e in EPS]; ss=[fn(e)[1] for e in EPS]
    ax.bar(x + (i-nb/2+0.5)*w, ms, w, yerr=ss, capsize=3, color=col, label=name)
ax.axhline(clean_ref, color="black", ls=":", lw=1.5, label=f"clean teacher ({clean_ref:.3f})")
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS])
ax.set_ylabel("vessel Dice (5 seeds, mean ± SEM)")
ax.set_ylim(0.55, 0.685)
ax.set_title("Mechanism comparison @ matched ε  (random-drop baseline PENDING — see note)")
ax.legend(fontsize=9, ncol=2, loc="upper left"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE/"fig_comparison.png", dpi=140)
print("saved fig_comparison.png")

# =====================================================================
# Print the best-config summary table
# =====================================================================
print("\n==== BEST CONFIG SUMMARY (same protocol, 5 seeds) ====")
print(f"clean teacher ref = {clean_ref:.4f}")
hdr=f"{'method':<26}" + "".join(f"  ε={int(e):<5}" for e in EPS)
print(hdr); print("-"*len(hdr))
for name,fn,_ in methods:
    print(f"{name:<26}" + "".join(f"  {fn(e)[0]:.4f}" for e in EPS))
print(f"{'PATE K=10 (pruning-file ref)':<26}" + "".join(f"  {prune['pate_k10_reference'][str(e)]:.4f}" for e in EPS))
# best pruning per eps
print("\nbest single-teacher pruning per ε:")
for e in EPS:
    best=max(prune['keep_fractions'], key=lambda f: prune['sweep'][str(e)][f'{f:.2f}']['mean'])
    c=prune['sweep'][str(e)][f'{best:.2f}']
    print(f"  ε={int(e):>2}:  keep {int(best*100)}% ({c['n_active']} ch)  Dice={c['mean']:.4f}  σ_active={c['sigma_active']:.2f}")