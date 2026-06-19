"""Ablation ladder: ONE pipeline, three orthogonal knobs.
  sensitivity K (PATE) | allocation uniform-vs-CANAL | dimensionality keep-frac
Shows which knob moves the needle; CANAL marked as the null knob.
All bars = TinyUNet single/multi-teacher, 5 seeds, same protocol.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
def L(n): return json.load(open(HERE / n))
ksat    = L("drive_pate_K_saturation_results.json")
compare = L("drive_compare_pate_canal_uniform_results.json")
combined= L("drive_pate_canal_combined_results.json")
joint   = L("drive_pate_pruning_joint_results.json")
abl     = L("drive_pruning_ablation_results.json")
EPS = [1.0, 2.0, 3.0, 4.0, 5.0]; SQ5 = np.sqrt(5)

floor   = abl["no_noise"]["student_only"]["vessel_dice"]["mean"]
ceiling = abl["no_noise"]["clean_full"]["vessel_dice"]["mean"]

def ks(K,e):
    c=ksat["sweep"][str(K)]["epsilons"][f"{e}"]; return c["mean"], c["sem"]
def canal_k5(e):
    c=combined["sweep"][str(e)]["CANAL"]; return c["mean"], c["std"]/SQ5
def joint_cell(K, f, e):
    # nesting is sweep[K][eps][keep]
    cell = joint["sweep"][str(K)][f"{e}"][f"{f:.2f}"]
    return cell["mean"], cell.get("sem", cell.get("std",0)/SQ5)

# ---- the ladder: branch off a fixed K=5 base so each knob toggles ALONE ----
# rungs 2-4 all at K=5 (only allocation / dimensionality differs); rung 5 cranks K to 10.
# (label, getter, color, is_null_knob)
def get_baseline(e): return joint_cell(1, 1.00, e)  # K=1 uniform full
def uni_k5(e):
    c=combined["sweep"][str(e)]["uniform"]; return c["mean"], c["std"]/SQ5
def get_pate(e):     return uni_k5(e)               # +PATE base: K=5 uniform full (SAME run as CANAL)
def get_canal(e):    return canal_k5(e)             # +CANAL: K=5 CANAL full  (toggle allocation only)
def get_prune(e):    return joint_cell(5, 0.02, e)  # +prune: K=5 uniform keep2% (toggle dim only)
def get_full(e):     return joint_cell(10, 0.02, e) # crank K=10 + keep2% = best

ceiling2 = abl["no_noise"]["clean_imp10"]["vessel_dice"]["mean"]  # fairer ceiling: clean + prune

rungs = [
    ("baseline\nK=1, uniform, full",       get_baseline, "#9e9e9e", False),
    ("+ PATE (base)\nK=5, uniform, full",  get_pate,     "#ff7f0e", False),
    ("+ CANAL\nK=5, CANAL, full",          get_canal,    "#1f77b4", True),
    ("+ prune\nK=5, uniform, keep2%",      get_prune,    "#8c564b", False),
    ("Joint best\nK=10, keep2%",           get_full,     "#2ca02c", False),
]
fig, ax = plt.subplots(figsize=(15, 6.8))
nb=len(rungs); w=0.15; x=np.arange(len(EPS))
canal_pos=None
for i,(name,fn,col,is_null) in enumerate(rungs):
    ms=[fn(e)[0] for e in EPS]; ss=[fn(e)[1] for e in EPS]
    pos=x+(i-nb/2+0.5)*w
    ax.bar(pos, ms, w, yerr=ss, capsize=3, color=col,
           hatch=("xx" if is_null else ""), edgecolor="black", linewidth=1.0, label=name)
    if is_null: canal_pos=pos
ax.axhline(floor,   color="black", ls="--", lw=1.6, label=f"FLOOR student-only (no teacher) {floor:.3f}")
ax.axhline(ceiling,  color="green", ls=":",  lw=1.6, label=f"clean teacher, full ch {ceiling:.3f}")
ax.axhline(ceiling2, color="purple",ls=(0,(5,1)), lw=1.6, label=f"clean teacher + prune (fairer ceiling) {ceiling2:.3f}")
if canal_pos is not None:
    # vertical callout in the CANAL bar's clear column (ε=2 group) — avoids overlapping tall neighbours
    yt=get_canal(EPS[0])[0]
    ax.annotate("CANAL ≈ uniform → null",
                xy=(canal_pos[0], yt+0.004), xytext=(canal_pos[0], 0.702),
                fontsize=8.5, ha="center", va="bottom", fontweight="bold", color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS], fontsize=12)
ax.set_ylabel("vessel Dice (5 seeds, mean ± SEM)")
ax.set_ylim(0.55, 0.72)
ax.set_title("Ablation ladder — one pipeline, three orthogonal knobs\n"
             "sensitivity K (PATE) ✓   ·   allocation uniform→CANAL ✗ (null)   ·   dimensionality keep-frac ✓",
             fontweight="bold")
ax.legend(fontsize=8.6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE/"fig_ablation_ladder.png", dpi=145, bbox_inches="tight")
print("saved fig_ablation_ladder.png")
print(f"floor={floor:.3f} ceiling={ceiling:.3f}")
for name,fn,_,_ in rungs:
    lab=name.split(chr(10))[0]
    print(f"  {lab:22s} " + " ".join(f"{fn(e)[0]:.3f}" for e in EPS))
