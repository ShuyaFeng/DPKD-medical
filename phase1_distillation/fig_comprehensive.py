"""Comprehensive comparison: our Joint-best vs DP baselines on DRIVE.
All bars: TinyUNet base-16 student, val vessel Dice, 5 seeds, per-patient ε.
Joint-source bars share one protocol; CANAL from the K=5 combined run; DP-SGD
trained directly on the 20 private images (Opacus RDP accountant)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
def L(n): return json.load(open(HERE / n))
joint    = L("drive_pate_pruning_joint_results.json")
combined = L("drive_pate_canal_combined_results.json")
abl      = L("drive_pruning_ablation_results.json")
EPS = [1.0, 2.0, 3.0, 4.0, 5.0]; SQ5 = np.sqrt(5)
try:
    dpsgd = L("drive_dpsgd_baseline_results.json"); DPSGD_PROVISIONAL = False
except FileNotFoundError:
    # PROVISIONAL placeholders while the full sweep runs (partial ε=2 seeds + ε=8 probe)
    dpsgd = {"sweep": {"2.0": {"mean": 0.108, "sem": 0.035},
                       "8.0": {"mean": 0.177, "sem": 0.000},
                       "16.0": {"mean": 0.200, "sem": 0.000}}}
    DPSGD_PROVISIONAL = True
try:
    inputdp = L("drive_input_dp_baseline_results.json"); IDP_PROVISIONAL = False
except FileNotFoundError:
    inputdp = {"sweep": {f"{e}": {"mean": 0.40, "sem": 0.0} for e in (2.0, 8.0, 16.0)}}
    IDP_PROVISIONAL = True
PROVISIONAL = DPSGD_PROVISIONAL or IDP_PROVISIONAL

floor   = abl["no_noise"]["student_only"]["vessel_dice"]["mean"]
ceiling = abl["no_noise"]["clean_full"]["vessel_dice"]["mean"]

def jc(K,f,e):  c=joint["sweep"][str(K)][f"{e}"][f"{f:.2f}"];           return c["mean"], c["sem"]
def canal(e):   c=combined["sweep"][str(e)]["CANAL"];                   return c["mean"], c["std"]/SQ5
def dps(e):     c=dpsgd["sweep"][f"{e}"];                               return c["mean"], c.get("sem",0)
def idp(e):     c=inputdp["sweep"][f"{e}"];                             return c["mean"], c.get("sem",0)

# (label, getter, color, family)
EXT="external DP baseline"; OURS="our pipeline (ablation)"; BEST="our method"
bars = [
    ("DP-SGD\n(student, no teacher)",   dps,                 "#d62728", EXT),
    ("DP feat-release\nK=1 uniform",    lambda e: jc(1,1.0,e),  "#9e9e9e", OURS),
    ("+ CANAL alloc\nK=5 (null)",       canal,               "#1f77b4", OURS),
    ("+ PATE\nK=10 uniform",            lambda e: jc(10,1.0,e), "#ff7f0e", OURS),
    ("+ prune\nK=1 keep2%",             lambda e: jc(1,0.02,e), "#8c564b", OURS),
    ("Joint best (OURS)\nK=10 keep2%",  lambda e: jc(10,0.02,e),"#2ca02c", BEST),
]
fig, ax = plt.subplots(figsize=(16, 6.8))
nb=len(bars); w=0.13; x=np.arange(len(EPS))
for i,(name,fn,col,fam) in enumerate(bars):
    ms=[fn(e)[0] for e in EPS]; ss=[fn(e)[1] for e in EPS]
    pos=x+(i-nb/2+0.5)*w
    hatch = "xx" if fam==EXT else ("//" if "null" in name else "")
    lw = 2.0 if fam==BEST else 1.0
    ax.bar(pos, ms, w, yerr=ss, capsize=2.5, color=col, hatch=hatch,
           edgecolor="black", linewidth=lw, label=name)
ax.axhline(floor,   color="black", ls="--", lw=1.6, label=f"no-teacher floor (=ε→∞ DP-SGD) {floor:.3f}")
ax.axhline(ceiling, color="green", ls=":",  lw=1.8, label=f"clean-teacher ceiling (non-private) {ceiling:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS], fontsize=13)
ax.set_ylabel("vessel Dice (5 seeds, mean ± SEM)")
ax.set_ylim(0.0, 0.74)
ax.set_title("Comprehensive comparison on DRIVE — our Joint method vs DP baselines\n"
             "(per-patient ε; DP-SGD trains the student directly under DP, Opacus RDP accountant)",
             fontweight="bold")
if PROVISIONAL:
    msg = "⚠ PROVISIONAL: " + ", ".join(
        [m for m,f in [("DP-SGD",DPSGD_PROVISIONAL),("input-DP",IDP_PROVISIONAL)] if f]) \
        + " still running"
    ax.text(0.5, 0.97, msg, transform=ax.transAxes, ha="center", va="top", fontsize=10,
            color="#d62728", fontweight="bold",
            bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d62728"))
ax.legend(fontsize=8.3, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)
ax.grid(alpha=0.3, axis="y")
fig.savefig(HERE/"fig_comprehensive.png", dpi=145, bbox_inches="tight")
print("saved fig_comprehensive.png")

print(f"\nfloor={floor:.3f} ceiling={ceiling:.3f}")
print(f"{'method':<28}" + "".join(f"  ε={int(e):<4}" for e in EPS))
for name,fn,_,fam in bars:
    lab=name.split(chr(10))[0]
    print(f"{lab:<28}" + "".join(f"  {fn(e)[0]:.3f}" for e in EPS) + f"   [{fam}]")
