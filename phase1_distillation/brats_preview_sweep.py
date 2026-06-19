"""
BraTS preview sweep (CLUSTER — needs real 4-modality data via prep_brats.py --src).
Mirrors the ISIC preview but with 4-channel input, and adds THE BraTS question:
does CANAL (channel water-filling) finally help when the channels are genuinely
heterogeneous (4 MRI modalities), unlike the flat RGB of DRIVE/ISIC?

Series @ ε∈{1,2,3,4,5}, 3 seeds:
  K=1 uniform · PATE K=10 · Joint K=10 keep2%       (headline, reproduces DRIVE)
  PATE K=5 uniform  vs  PATE K=5 + CANAL            (the multi-modal CANAL test)
+ floor (no teacher) / ceiling (clean distill).
Writes brats_preview_results.json + fig_brats_preview.png.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from brats_dataset import BRATSDataset
from drive_local_demo import TinyUNet, train_teacher, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma, precompute_joint_cache
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
IN_CH = 4
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}  (BraTS, in_ch={IN_CH})")
train_ds, val_ds = BRATSDataset("train", 96), BRATSDataset("val", 96)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f"BraTS: train={len(train_ds)}  val={len(val_ds)}")
EPS, SEEDS, TE, SE = [1.0, 2.0, 3.0, 4.0, 5.0], [100, 200, 300], 50, 40

tk = {K: train_K_teachers(train_ds, K, device, n_epochs=TE, in_ch=IN_CH) for K in (1, 5, 10)}
Cb = tk[1][0][0].base * 4
imp = {K: shared_importance(tk[K][0], DataLoader(train_ds, batch_size=8), device).to(device) for K in (5, 10)}
rank10 = torch.argsort(imp[10], descending=True)


def run_students(cache):
    return [train_student_distill(train_ds, val_loader, cache, device, student_base=16,
                                  teacher_base=32, n_epochs=SE, lr=1e-3, lambda_feat=0.4,
                                  seed=s, in_ch=IN_CH)[0] for s in SEEDS]


def series_cells(K, frac=1.0, canal=False):
    teachers, caps_list = tk[K]
    deltas = torch.full((Cb,), 2.0 / K, device=device)
    am = torch.zeros(Cb, dtype=torch.bool, device=device)
    am[(rank10 if K == 10 else torch.argsort(imp.get(K, imp[10]), descending=True))[:max(1, int(round(frac * Cb)))]] = True
    out = {}
    for e in EPS:
        rho = eps_to_rho(e)
        sig = correct_waterfilling_sigma(deltas, imp[K], rho) if canal else thresholded_uniform_sigma(deltas, rho, am)
        if canal:  # CANAL allocates over ALL channels
            am_use = torch.ones(Cb, dtype=torch.bool, device=device)
        else:
            am_use = am
        cache = precompute_joint_cache(teachers, caps_list, train_ds, sig, am_use, device, seed=42 + int(e))
        d = run_students(cache)
        out[f"{e}"] = {"dices": d, "mean": float(np.mean(d)), "sem": float(np.std(d) / np.sqrt(len(d)))}
        print(f"   K={K} frac={frac} canal={canal} ε={e}: {np.mean(d):.4f}")
    return out


results = {"in_ch": IN_CH, "train": len(train_ds), "epsilons": EPS, "seeds": SEEDS, "series": {}}
results["series"]["K=1 uniform"]      = series_cells(1, 1.0)
results["series"]["PATE K=10"]        = series_cells(10, 1.0)
results["series"]["Joint K=10 keep2%"] = series_cells(10, 0.02)
results["series"]["PATE K=5 uniform"] = series_cells(5, 1.0)
results["series"]["PATE K=5 + CANAL"] = series_cells(5, 1.0, canal=True)   # THE multi-modal CANAL test

floor_d = []
for s in SEEDS:
    torch.manual_seed(s)
    stu = TinyUNet(in_ch=IN_CH, num_classes=2, base=16).to(device)
    train_teacher(stu, DataLoader(train_ds, batch_size=8, shuffle=True), n_epochs=SE, lr=1e-3, device=device)
    floor_d.append(evaluate_vessel_dice(stu, val_loader, device))
clean = precompute_joint_cache(tk[1][0], tk[1][1], train_ds, torch.zeros(Cb, device=device),
                               torch.ones(Cb, dtype=torch.bool, device=device), device, seed=7)
results["floor"] = {"mean": float(np.mean(floor_d)), "dices": floor_d}
results["ceiling"] = {"mean": float(np.mean(run_students(clean)))}
(HERE / "brats_preview_results.json").write_text(json.dumps(results, indent=2))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(EPS)); names = list(results["series"]); nb = len(names); w = 0.15
for i, n in enumerate(names):
    ms = [results["series"][n][f"{e}"]["mean"] for e in EPS]
    ss = [results["series"][n][f"{e}"]["sem"] for e in EPS]
    ax.bar(x + (i - nb / 2 + 0.5) * w, ms, w, yerr=ss, capsize=2.5, edgecolor="black", label=n)
ax.axhline(results["floor"]["mean"], color="black", ls="--", lw=1.5, label=f"floor {results['floor']['mean']:.3f}")
ax.axhline(results["ceiling"]["mean"], color="green", ls=":", lw=1.7, label=f"ceiling {results['ceiling']['mean']:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS])
ax.set_ylabel("tumor Dice (3 seeds, mean ± SEM)")
ax.set_title(f"BraTS (multi-modal) — n={len(train_ds)}: mechanism + the CANAL test on 4 heterogeneous channels", fontweight="bold")
ax.legend(fontsize=8.5, ncol=2, loc="lower right"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(HERE / "fig_brats_preview.png", dpi=130)
print("saved brats_preview_results.json + fig_brats_preview.png")
