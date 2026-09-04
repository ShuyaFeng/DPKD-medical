"""
Standalone script: regenerate figs/abl_k.png from
results/isic_K_eps_sweep_arxiv_results.json.

No GPU needed — reads the precomputed JSON and plots K vs Dice at eps=8.
Uniform allocation only, no error bars, matching paper Figure 3 style.
Expects K=[1,3,5,10,16], seeds=3, eps=8 from run_arxiv_fig3.sh.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE      = Path(__file__).parent
JSON_PATH = HERE / "results" / "isic_K_eps_sweep_arxiv_results.json"
OUT_PATH  = HERE.parent / "WACV26" / "figs" / "abl_k.png"

data = json.loads(JSON_PATH.read_text())

eps = "8.0"
Ks  = data["Ks"]   # [1, 3, 5, 10, 16]

uniform_means = [data["series"][f"PATE+uniform (K={K})"][eps]["mean"] for K in Ks]

fig, ax = plt.subplots(figsize=(5.5, 5))

ax.plot(
    Ks, [m * 100 for m in uniform_means],
    color="#1f77b4", ls="-", marker="o",
    lw=2.5, ms=12,
)

ax.set_xticks(Ks)
ax.set_xticklabels([str(k) for k in Ks], fontsize=18)
ax.set_xlabel("Ensemble size $K$", fontsize=20)
ax.set_ylabel("Dice (%)", fontsize=20)
ax.set_ylim(81.5, 85.3)
ax.set_yticks([82, 83, 84, 85])
ax.tick_params(axis="y", labelsize=18)
ax.grid(alpha=0.3)
fig.tight_layout()

fig.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")

# Print summary table
print(f"\n{'K':>4}  {'Uniform (%)':>12}")
for K, m in zip(Ks, uniform_means):
    print(f"{K:>4}  {m*100:.2f}")
