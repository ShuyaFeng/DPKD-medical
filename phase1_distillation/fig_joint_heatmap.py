"""Joint K x keep heatmap, one panel per epsilon. The 'paper heatmap'.
Color = student vessel Dice; cells at/above clean ceiling boxed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
joint = json.load(open(HERE / "drive_pate_pruning_joint_results.json"))
abl   = json.load(open(HERE / "drive_pruning_ablation_results.json"))
floor   = abl["no_noise"]["student_only"]["vessel_dice"]["mean"]
ceiling = abl["no_noise"]["clean_full"]["vessel_dice"]["mean"]

sw = joint["sweep"]
Ks   = ["1", "5", "10"]
keeps= ["1.00", "0.30", "0.20", "0.10", "0.05", "0.02"]
nact = {k: sw["1"]["2.0"][k]["n_active"] for k in keeps}
EPS  = ["2.0", "8.0", "16.0"]

vmin = min(sw[K][e][k]["mean"] for K in Ks for e in EPS for k in keeps)
vmax = max(sw[K][e][k]["mean"] for K in Ks for e in EPS for k in keeps)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, e in zip(axes, EPS):
    M = np.array([[sw[K][e][k]["mean"] for k in keeps] for K in Ks])  # rows=K, cols=keep
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(keeps)))
    ax.set_xticklabels([f"{int(float(k)*100)}%\n{nact[k]}ch" for k in keeps], fontsize=8)
    ax.set_yticks(range(len(Ks))); ax.set_yticklabels([f"K={K}" for K in Ks])
    ax.set_xlabel("channels kept"); ax.set_title(f"ε = {int(float(e))}")
    for i,K in enumerate(Ks):
        for j,k in enumerate(keeps):
            v = M[i,j]
            txt = f"{v:.3f}"
            mark = "★" if v >= ceiling else ("·" if v < floor else "")
            ax.text(j, i, txt+("\n"+mark if mark else ""), ha="center", va="center",
                    fontsize=7.5, color="white" if v < (vmin+vmax)/2 else "black", fontweight="bold")
            if v >= ceiling:   # box cells at/above clean ceiling
                ax.add_patch(Rectangle((j-0.5,i-0.5),1,1, fill=False, edgecolor="red", lw=2))
axes[0].set_ylabel("teachers K (sensitivity 2/K)")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("vessel Dice (5 seeds)")
fig.suptitle(f"Joint PATE×pruning sweep — red box = ≥ clean ceiling ({ceiling:.3f}); '·' = below no-teacher floor ({floor:.3f})",
             fontweight="bold", y=1.02)
fig.savefig(HERE/"fig_joint_heatmap.png", dpi=140, bbox_inches="tight")
print("saved fig_joint_heatmap.png")
