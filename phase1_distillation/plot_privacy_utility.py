"""
Privacy-utility tradeoff figure for channel-WF vs uniform DP noise.

Pulls together three data sources already in the repo:
  1. DRIVE noise-only probe   (results/phase1_channel_noise_summary_v2.json)
  2. DRIVE GKD-v2 distillation (results/GKD_V2_*_summary.json, 3 seeds each)
  3. Synthetic local demo      (synthetic_demo_results.json)

Produces phase1_distillation/privacy_utility_tradeoff.png
"""

import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# ---------------------------------------------------------------------------
# 1. Load DRIVE noise-only probe
# ---------------------------------------------------------------------------
probe = json.load(open(RESULTS / "phase1_channel_noise_summary_v2.json"))
probe_eps = [float(e) for e in probe["epsilons"]]
probe_clean = probe["clean_dice"]
probe_uni = [probe["mean_dice"]["uniform"][str(e)] for e in probe["epsilons"]]
probe_wf  = [probe["mean_dice"]["channel_WF"][str(e)] for e in probe["epsilons"]]
probe_lift = [w - u for w, u in zip(probe_wf, probe_uni)]


# ---------------------------------------------------------------------------
# 2. Load DRIVE GKD-v2 distillation, aggregate over seeds
# ---------------------------------------------------------------------------
agg = defaultdict(list)
for f in sorted(glob.glob(str(RESULTS / "GKD_V2_*_summary.json"))):
    d = json.load(open(f))
    agg[(d["noise_type"], float(d["epsilon"]))].append(d["best_mDice"])

distill_eps = sorted({k[1] for k in agg})
student_baseline = 0.8791


def mean_std(nt):
    means, stds, ses = [], [], []
    for e in distill_eps:
        vals = agg[(nt, e)]
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        means.append(m)
        stds.append(s)
        ses.append(s / math.sqrt(len(vals)))   # standard error of the mean
    return means, stds, ses


wf_m,  wf_s,  wf_se  = mean_std("channel_WF")
uni_m, uni_s, uni_se = mean_std("uniform")
distill_lift = [w - u for w, u in zip(wf_m, uni_m)]
# SE of the difference of two independent means
distill_lift_se = [math.sqrt(a * a + b * b) for a, b in zip(wf_se, uni_se)]


# ---------------------------------------------------------------------------
# 3. Load synthetic demo probe
# ---------------------------------------------------------------------------
syn = json.load(open(HERE / "synthetic_demo_results.json"))
syn_eps = sorted(float(e) for e in syn["noise_probe"]["uniform"])
syn_clean = syn["noise_probe"]["no_noise"]
syn_uni = [syn["noise_probe"]["uniform"][str(e)] for e in syn_eps]
syn_wf  = [syn["noise_probe"]["channel_WF"][str(e)] for e in syn_eps]


# ---------------------------------------------------------------------------
# Plot: 2x2 grid
# ---------------------------------------------------------------------------
C_WF  = "#1f77b4"   # channel-WF  = blue
C_UNI = "#d62728"   # uniform     = red
C_REF = "#555555"

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Channel-WF Privacy–Utility Tradeoff  (lower $\\epsilon$ = stronger privacy)",
             fontsize=15, fontweight="bold")

# --- (a) DRIVE noise-only probe ----------------------------------------------
ax = axes[0, 0]
ax.plot(probe_eps, probe_wf,  "o-", color=C_WF,  lw=2, ms=7, label="channel-WF")
ax.plot(probe_eps, probe_uni, "s--", color=C_UNI, lw=2, ms=6, label="uniform")
ax.axhline(probe_clean, color=C_REF, ls=":", lw=1.5,
           label=f"clean / no noise ({probe_clean:.3f})")
ax.set_xscale("log", base=2)
ax.set_xticks(probe_eps)
ax.set_xticklabels([str(int(e)) for e in probe_eps])
ax.set_xlabel("privacy budget  $\\epsilon$")
ax.set_ylabel("vessel Dice")
ax.set_title("(a) DRIVE — noise-only probe (no training)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

# --- (b) DRIVE GKD-v2 distillation -------------------------------------------
ax = axes[0, 1]
ax.errorbar(distill_eps, wf_m, yerr=wf_s, fmt="o-", color=C_WF, lw=2, ms=7,
            capsize=4, label="channel-WF  (mean$\\pm$std, 3 seeds)")
ax.errorbar(distill_eps, uni_m, yerr=uni_s, fmt="s--", color=C_UNI, lw=2, ms=6,
            capsize=4, label="uniform  (mean$\\pm$std, 3 seeds)")
ax.axhline(student_baseline, color=C_REF, ls=":", lw=1.5,
           label=f"student baseline ({student_baseline:.3f})")
ax.set_xscale("log", base=2)
ax.set_xticks(distill_eps)
ax.set_xticklabels([str(int(e)) for e in distill_eps])
ax.set_xlabel("privacy budget  $\\epsilon$")
ax.set_ylabel("mDice (mean over classes)")
ax.set_title("(b) DRIVE — after GKD-v2 distillation")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

# --- (c) WF - uniform lift ---------------------------------------------------
ax = axes[1, 0]
ax.axhline(0, color="black", lw=1)
ax.plot(probe_eps, probe_lift, "o-", color="#2ca02c", lw=2, ms=7,
        label="noise-only probe")
ax.errorbar(distill_eps, distill_lift, yerr=distill_lift_se, fmt="D-",
            color="#9467bd", lw=2, ms=6, capsize=4,
            label="GKD-v2 distillation ($\\pm$SE of diff)")
ax.fill_between([min(probe_eps) * 0.8, max(probe_eps) * 1.2], -0.002, 0.002,
                color="gray", alpha=0.15, label="$\\pm$0.002 (seed noise band)")
ax.set_xscale("log", base=2)
ax.set_xticks(probe_eps)
ax.set_xticklabels([str(int(e)) for e in probe_eps])
ax.set_xlim(min(probe_eps) * 0.8, max(probe_eps) * 1.2)
ax.set_xlabel("privacy budget  $\\epsilon$")
ax.set_ylabel("Dice lift:  channel-WF $-$ uniform")
ax.set_title("(c) Does channel-WF beat uniform?")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.3)

# --- (d) Synthetic local demo ------------------------------------------------
ax = axes[1, 1]
ax.plot(syn_eps, syn_wf,  "o-", color=C_WF,  lw=2, ms=7, label="channel-WF")
ax.plot(syn_eps, syn_uni, "s--", color=C_UNI, lw=2, ms=6, label="uniform")
ax.axhline(syn_clean, color=C_REF, ls=":", lw=1.5,
           label=f"clean / no noise ({syn_clean:.3f})")
ax.set_xscale("log", base=2)
ax.set_xticks(syn_eps)
ax.set_xticklabels([str(e) for e in syn_eps])
ax.set_xlabel("privacy budget  $\\epsilon$")
ax.set_ylabel("IoU")
ax.set_title("(d) Synthetic demo — noise-only probe (local run)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = HERE / "privacy_utility_tradeoff.png"
fig.savefig(out, dpi=150)
print(f"Saved figure to {out}")

# ---------------------------------------------------------------------------
# Also print the numbers driving the figure
# ---------------------------------------------------------------------------
print("\n=== (a) DRIVE noise-only probe ===")
print(f"  clean vessel Dice = {probe_clean:.4f}")
for e, u, w, l in zip(probe_eps, probe_uni, probe_wf, probe_lift):
    print(f"  eps={e:>5.1f}  uniform={u:.4f}  WF={w:.4f}  lift={l:+.4f}")

print("\n=== (b) DRIVE GKD-v2 distillation (3 seeds) ===")
print(f"  student baseline mDice = {student_baseline:.4f}")
for i, e in enumerate(distill_eps):
    print(f"  eps={e:>5.1f}  uniform={uni_m[i]:.4f}+-{uni_s[i]:.4f}  "
          f"WF={wf_m[i]:.4f}+-{wf_s[i]:.4f}  "
          f"lift={distill_lift[i]:+.4f}+-{distill_lift_se[i]:.4f}")
