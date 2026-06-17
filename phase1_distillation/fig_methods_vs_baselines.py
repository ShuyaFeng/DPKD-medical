"""Ours-vs-baseline comparison, CANAL explicitly marked.
All bars share the same TinyUNet single-teacher 5-seed protocol."""
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
prune   = L("drive_single_teacher_pruning_results.json")
abl     = L("drive_pruning_ablation_results.json")
EPS = [2.0, 8.0, 16.0]
SQ5 = np.sqrt(5)

def ks(K,e):  c=ksat["sweep"][str(K)]["epsilons"][f"{e}"];        return c["mean"], c["sem"]
def cn(e):    c=compare["K1_CANAL"][str(e)];                       return c["mean"], c["sem"]
def un(e):    c=compare["K1_uniform"][str(e)];                     return c["mean"], c["sem"]
def pr(f,e):  c=prune["sweep"][str(e)][f"{f:.2f}"];               return c["mean"], c["sem"]
def rnd(e):   c=abl["noisy"][str(e)]["rand_keep02"]["vessel_dice"];return c["mean"], c["std"]/SQ5

floor   = abl["no_noise"]["student_only"]["vessel_dice"]["mean"]
ceiling = abl["no_noise"]["clean_full"]["vessel_dice"]["mean"]

# (label, fn, color, hatch, group)
BASE = "BASELINE"; OURS = "OURS"
bars = [
    ("uniform, K=1 (no mechanism)", un,            "#9e9e9e", "//", BASE),
    ("random drop, keep 2%",        rnd,           "#cfcfcf", "xx", BASE),
    ("CANAL  (channel WF alloc)",   cn,            "#1f77b4", "",   OURS),
    ("PATE  K=5",                   lambda e:ks(5,e),  "#ffbb78", "", OURS),
    ("PATE  K=10",                  lambda e:ks(10,e), "#ff7f0e", "", OURS),
    ("prune  keep 5% (importance)", lambda e:pr(0.05,e),"#98df8a", "", OURS),
    ("prune  keep 2% (importance)", lambda e:pr(0.02,e),"#2ca02c", "", OURS),
]
fig, ax = plt.subplots(figsize=(13, 6.6))
nb=len(bars); w=0.115; x=np.arange(len(EPS))
canal_xs=[]
for i,(name,fn,col,hatch,grp) in enumerate(bars):
    ms=[fn(e)[0] for e in EPS]; ss=[fn(e)[1] for e in EPS]
    pos=x+(i-nb/2+0.5)*w
    edge = "black" if grp==OURS else "#555555"
    ax.bar(pos, ms, w, yerr=ss, capsize=2.5, color=col, hatch=hatch,
           edgecolor=edge, linewidth=1.1 if grp==OURS else 0.8,
           label=("[ours] " if grp==OURS else "[base] ")+name)
    if name.startswith("CANAL"): canal_xs=pos
# reference lines
ax.axhline(floor,   color="black", ls="--", lw=1.6, label=f"[base] student-only FLOOR (no teacher) {floor:.3f}")
ax.axhline(ceiling, color="green", ls=":",  lw=1.8, label=f"[base] clean-teacher CEILING {ceiling:.3f}")
# annotate CANAL
ytop=cn(EPS[2])[0]
ax.annotate("CANAL = our allocation\n≈ uniform  →  NULL result",
            xy=(canal_xs[2], ytop), xytext=(canal_xs[2]-0.15, ytop+0.045),
            fontsize=9, ha="center", fontweight="bold", color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))

ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS], fontsize=12)
ax.set_ylabel("vessel Dice (single teacher K=1 unless noted, 5 seeds ± SEM)")
ax.set_ylim(0.55, 0.70)
ax.set_title("Our methods (colored, black edge) vs baselines (gray, hatched) on DRIVE\n"
             "parameters: CANAL alloc · PATE K∈{5,10} · pruning keep∈{5%,2%}", fontweight="bold")
ax.legend(fontsize=8.2, ncol=2, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE/"fig_methods_vs_baselines.png", dpi=145)
print("saved fig_methods_vs_baselines.png")
print(f"floor={floor:.3f} ceiling={ceiling:.3f}")
for name,fn,_,_,grp in bars:
    print(f"  [{grp:8s}] {name:30s} " + " ".join(f"{fn(e)[0]:.3f}" for e in EPS))
