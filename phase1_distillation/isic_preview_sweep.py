"""Lightweight preliminary ISIC sweep (200-img subset) — does the DRIVE story
reproduce? Headline series at ε∈{1,2,3,4,5}, 3 seeds, + floor/ceiling.
NOT a final result (subset). Writes isic_preview_results.json + fig_isic_preview.png."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from isic_dataset import ISICDataset
from drive_local_demo import TinyUNet, train_teacher, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma, precompute_joint_cache
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
train_ds, val_ds = ISICDataset("train", 96), ISICDataset("val", 96)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
EPS = [1.0, 2.0, 3.0, 4.0, 5.0]; SEEDS = [100, 200, 300]
TE, SE = 40, 30   # teacher / student epochs

# teachers for K=1 and K=10 (trained once)
tk = {}
for K in (1, 10):
    print(f"training {K} teacher(s)...")
    tk[K] = train_K_teachers(train_ds, K, device, n_epochs=TE)
Cb = tk[1][0][0].base * 4
imp = shared_importance(tk[10][0], DataLoader(train_ds, batch_size=8), device).to(device)
rank = torch.argsort(imp, descending=True)
def mask(frac):
    m = torch.zeros(Cb, dtype=torch.bool, device=device)
    m[rank[:max(1, int(round(frac * Cb)))]] = True
    return m

def run_students(cache, n=len(SEEDS)):
    ds = []
    for s in SEEDS:
        best, _ = train_student_distill(train_ds, val_loader, cache, device,
                                        student_base=16, teacher_base=32,
                                        n_epochs=SE, lr=1e-3, lambda_feat=0.4, seed=s)
        ds.append(best)
    return ds

def cells(K, frac):
    teachers, caps_list = tk[K]
    am = mask(frac); deltas = torch.full((Cb,), 2.0 / K, device=device)
    out = {}
    for e in EPS:
        sig = thresholded_uniform_sigma(deltas, eps_to_rho(e), am)
        cache = precompute_joint_cache(teachers, caps_list, train_ds, sig, am, device, seed=42 + int(e))
        d = run_students(cache); out[f"{e}"] = {"dices": d, "mean": float(np.mean(d)), "sem": float(np.std(d) / np.sqrt(len(d)))}
        print(f"   K={K} keep={frac} ε={e}: {np.mean(d):.4f}")
    return out

results = {"subset": len(train_ds), "epsilons": EPS, "seeds": SEEDS, "series": {}}
print("\n[series] K=1 uniform full"); results["series"]["K=1 uniform"] = cells(1, 1.0)
print("[series] PATE K=10 full");    results["series"]["PATE K=10"]   = cells(10, 1.0)
print("[series] Joint K=10 keep2%"); results["series"]["Joint K=10 keep2%"] = cells(10, 0.02)

# floor (no teacher) + ceiling (clean K=1 full distill)
print("[ref] floor (student-only)")
floor_d = []
for s in SEEDS:
    torch.manual_seed(s)
    stu = TinyUNet(in_ch=3, num_classes=2, base=16).to(device)
    train_teacher(stu, DataLoader(train_ds, batch_size=8, shuffle=True), n_epochs=SE, lr=1e-3, device=device)
    floor_d.append(evaluate_vessel_dice(stu, val_loader, device))
print("[ref] ceiling (clean K=1 distill)")
am_all = torch.ones(Cb, dtype=torch.bool, device=device)
clean_cache = precompute_joint_cache(tk[1][0], tk[1][1], train_ds, torch.zeros(Cb, device=device), am_all, device, seed=7)
ceil_d = run_students(clean_cache)
results["floor"] = {"mean": float(np.mean(floor_d)), "dices": floor_d}
results["ceiling"] = {"mean": float(np.mean(ceil_d)), "dices": ceil_d}
(HERE / "isic_preview_results.json").write_text(json.dumps(results, indent=2))

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(EPS)); series = list(results["series"]); nb = len(series); w = 0.22
cols = {"K=1 uniform": "#9e9e9e", "PATE K=10": "#ff7f0e", "Joint K=10 keep2%": "#2ca02c"}
for i, name in enumerate(series):
    ms = [results["series"][name][f"{e}"]["mean"] for e in EPS]
    ss = [results["series"][name][f"{e}"]["sem"] for e in EPS]
    ax.bar(x + (i - nb / 2 + 0.5) * w, ms, w, yerr=ss, capsize=3, color=cols.get(name, "#1f77b4"),
           edgecolor="black", linewidth=1.0 if "Joint" in name else 0.8, label=name)
ax.axhline(results["floor"]["mean"], color="black", ls="--", lw=1.6, label=f"floor (no teacher) {results['floor']['mean']:.3f}")
ax.axhline(results["ceiling"]["mean"], color="green", ls=":", lw=1.8, label=f"clean-teacher ceiling {results['ceiling']['mean']:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS], fontsize=12)
ax.set_ylabel("lesion Dice (3 seeds, mean ± SEM)")
ax.set_title(f"ISIC 2018 — PRELIMINARY (subset n={len(train_ds)}): does the DRIVE story reproduce?", fontweight="bold")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE / "fig_isic_preview.png", dpi=130)
print("\nsaved isic_preview_results.json + fig_isic_preview.png")
print(f"floor={results['floor']['mean']:.3f}  ceiling={results['ceiling']['mean']:.3f}")
