"""
Unified DP comparison runner for ANY dataset (DRIVE / ISIC / BraTS).
MODIFIED: K=3 added to the teacher ablation; keep% swept across
{70,50,10,5,2}% for subsample/Joint series (was hardcoded to 2%);
headline subsample/Joint/PATE-K3 series now use RANDOM channel
selection (team decision: importance ~ random, use random as the
primary method, importance kept only for the dedicated control test).

Reads whatever is currently in data/<DATASET>_HF/ (local subset or full set),
auto-selects in_ch (DRIVE/ISIC=3 RGB, BraTS=4 modalities), and runs the full
comparison matrix in one go:
  refs:     floor (student-only, no teacher)   ceiling (clean-teacher distill)
  headline: K=1 uniform · PATE K=3/5/10 uniform · subsample K=1 (random, multi-keep%)
            · Joint K=10 (random, multi-keep%)
  control:  importance- vs RANDOM-selection at keep 2%        (selection is data-independent?)
  CANAL:    PATE K=5 uniform vs PATE K=5 + water-filling       (does allocation help?)
Saves results/<dataset>_comparison.json + fig_<dataset>_comparison.png.
Build data first:
  DRIVE: already in data/DRIVE_HF      ISIC: python download_isic.py --n 0
  BraTS: python prep_brats.py --src <root>
Examples:
  cluster full:  python run_comparison.py --dataset isic --seeds 5 --te 60 --se 40 --epsilons "2,4,8,16"
"""
import argparse, sys, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent))
from drive_local_demo import TinyUNet, train_teacher, evaluate_vessel_dice
from drive_pate_poc import train_K_teachers
from drive_pate_pruning_joint import shared_importance, thresholded_uniform_sigma, precompute_joint_cache
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)

# K values for the teacher ablation (Dr. Feng: K∈{1,3,5,10})
K_VALUES = (1, 3, 5, 10)
# K values that need an importance ranking (for CANAL water-filling test)
K_FOR_IMPORTANCE = (3, 5, 10)
# Keep fractions to sweep for subsample / Joint series (Dr. Tian: vary the keep%)
KEEP_FRACS = (0.70, 0.50, 0.10, 0.05, 0.02)
# Total "cells" (epsilon-points) across the whole comparison matrix, for progress/ETA
TOTAL_CELLS = 96  # 4(K=1,3,5,10 uniform) + 5*4*3(keep% sweeps) + 3*4(CANAL) + 2*4(control)

def get_dataset(name, split, size):
    if name == "drive":
        from drive_local_demo import DriveDataset; return DriveDataset(split, size), 3
    if name == "isic":
        from isic_dataset import ISICDataset; return ISICDataset(split, size), 3
    if name == "kvasir":
        from kvasir_dataset import KvasirDataset; return KvasirDataset(split, size), 3
    if name == "busi":
        from busi_dataset import BUSIDataset; return BUSIDataset(split, size), 1
    if name == "brats":
        from brats_dataset import BRATSDataset; return BRATSDataset(split, size), 4
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["drive", "isic", "kvasir", "busi", "brats"])
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilons", type=str, default="2,4,8,16")
    ap.add_argument("--te", type=int, default=50, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    ap.add_argument("--no-canal", action="store_true")
    ap.add_argument("--no-random", action="store_true")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds, in_ch = get_dataset(args.dataset, "train", args.size)
    val_ds, _ = get_dataset(args.dataset, "val", args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[{args.dataset}] device={dev} in_ch={in_ch}  train={len(train_ds)} val={len(val_ds)}  "
          f"ε={EPS} seeds={SEEDS}  K={K_VALUES}  keep%={KEEP_FRACS}")

    # teachers for K=1,3,5,10 (trained once, reused)
    tk = {K: train_K_teachers(train_ds, K, dev, n_epochs=args.te, in_ch=in_ch) for K in K_VALUES}
    Cb = tk[1][0][0].base * 4

    # importance ranking (needed only for CANAL water-filling + the control test)
    imp = {K: shared_importance(tk[K][0], DataLoader(train_ds, batch_size=8), dev).to(dev)
           for K in K_FOR_IMPORTANCE}

    def importance_mask(K, frac):
        m = torch.zeros(Cb, dtype=torch.bool, device=dev)
        r = torch.argsort(imp[K] if K in imp else imp[10], descending=True)
        m[r[:max(1, int(round(frac * Cb)))]] = True
        return m

    def rand_mask(frac, seed):
        g = torch.Generator().manual_seed(seed)
        m = torch.zeros(Cb, dtype=torch.bool, device=dev)
        idx = torch.randperm(Cb, generator=g)[:max(1, int(round(frac * Cb)))].to(dev)
        m[idx] = True
        return m

    def students(cache):
        return [train_student_distill(train_ds, val_loader, cache, dev, student_base=16,
                                      teacher_base=32, n_epochs=args.se, lr=1e-3,
                                      lambda_feat=0.4, seed=s, in_ch=in_ch)[0] for s in SEEDS]

    job_start = time.time()
    cell_counter = {"done": 0}

    def cell(K, am, sigma):
        cache = precompute_joint_cache(tk[K][0], tk[K][1], train_ds, sigma, am, dev, seed=42)
        d = students(cache)
        cell_counter["done"] += 1
        elapsed = time.time() - job_start
        pct = cell_counter["done"] / TOTAL_CELLS
        eta = elapsed / pct - elapsed if pct > 0 else 0
        print(f"   [progress] {cell_counter['done']}/{TOTAL_CELLS} cells "
              f"({pct*100:.1f}%) | elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min")
        return {"dices": d, "mean": float(np.mean(d)), "sem": float(np.std(d) / np.sqrt(len(d)))}

    def sweep(K, frac, canal=False, random_select=False, rand_seed=12345):
        """random_select=True uses RANDOM channel choice (the team's primary
        method for headline subsample/Joint series, since importance ~ random).
        canal=True ignores frac (always uses all channels + water-filling)."""
        deltas = torch.full((Cb,), 2.0 / K, device=dev)
        out = {}
        for e in EPS:
            rho = eps_to_rho(e)
            if canal:
                am = torch.ones(Cb, dtype=torch.bool, device=dev)
                sigma = correct_waterfilling_sigma(deltas, imp[K], rho)
            else:
                am = rand_mask(frac, rand_seed) if random_select else importance_mask(K, frac)
                sigma = thresholded_uniform_sigma(deltas, rho, am)
            out[f"{e}"] = cell(K, am, sigma)
            print(f"   {args.dataset} K={K} frac={frac} canal={canal} "
                  f"random={random_select} ε={e}: {out[f'{e}']['mean']:.4f}")
        return out

    R = {"dataset": args.dataset, "in_ch": in_ch, "train": len(train_ds), "epsilons": EPS,
         "seeds": SEEDS, "K_values": K_VALUES, "keep_fracs": KEEP_FRACS, "series": {}}

    results_path = HERE / "results" / f"{args.dataset}_comparison.json"

    def save_results(tag=""):
        results_path.write_text(json.dumps(R, indent=2))
        print(f"   [saved] results/{args.dataset}_comparison.json  {('(' + tag + ')') if tag else ''}")

    print("\n[headline: uniform, all channels]")
    R["series"]["K=1 uniform"]  = sweep(1, 1.0)
    R["series"]["PATE K=3 uniform"]  = sweep(3, 1.0)
    R["series"]["PATE K=5 uniform"]  = sweep(5, 1.0)
    R["series"]["PATE K=10 uniform"] = sweep(10, 1.0)
    save_results("after headline uniform")

    print("\n[headline: subsample K=1, random channels, swept keep%]")
    R["series"]["subsample K=1"] = {}
    for kf in KEEP_FRACS:
        R["series"]["subsample K=1"][f"keep{int(kf*100)}%"] = sweep(1, kf, random_select=True)
    save_results("after subsample K=1")

    print("\n[headline: Joint K=10, random channels, swept keep%]")
    R["series"]["Joint K=10"] = {}
    for kf in KEEP_FRACS:
        R["series"]["Joint K=10"][f"keep{int(kf*100)}%"] = sweep(10, kf, random_select=True)
    save_results("after Joint K=10")

    print("\n[headline: Joint K=3, random channels, swept keep%]  (K=3 ablation point)")
    R["series"]["Joint K=3"] = {}
    for kf in KEEP_FRACS:
        R["series"]["Joint K=3"][f"keep{int(kf*100)}%"] = sweep(3, kf, random_select=True)
    save_results("after Joint K=3")

    if not args.no_canal:
        print("\n[CANAL test: uniform vs water-filling, all channels]")
        R["series"]["PATE K=5 + CANAL"] = sweep(5, 1.0, canal=True)
        R["series"]["PATE K=3 + CANAL"] = sweep(3, 1.0, canal=True)
        R["series"]["PATE K=10 + CANAL"] = sweep(10, 1.0, canal=True)
        save_results("after CANAL test")

    if not args.no_random:
        print("\n[selection control: importance vs random @ keep2%, K=10]")
        deltas10 = torch.full((Cb,), 2.0 / 10, device=dev)
        R["random_control"] = {}
        for e in EPS:
            rho = eps_to_rho(e)
            am_i = importance_mask(10, 0.02); am_r = rand_mask(0.02, 12345)
            ri = cell(10, am_i, thresholded_uniform_sigma(deltas10, rho, am_i))
            rr = cell(10, am_r, thresholded_uniform_sigma(deltas10, rho, am_r))
            R["random_control"][f"{e}"] = {"importance": ri, "random": rr}
            print(f"   ε={e}: imp={ri['mean']:.4f}  rand={rr['mean']:.4f}")
        save_results("after random control")

    print("\n[refs] floor + ceiling")
    floor_d = []
    for s in SEEDS:
        torch.manual_seed(s)
        stu = TinyUNet(in_ch=in_ch, num_classes=2, base=16).to(dev)
        train_teacher(stu, DataLoader(train_ds, batch_size=8, shuffle=True), n_epochs=args.se, lr=1e-3, device=dev)
        floor_d.append(evaluate_vessel_dice(stu, val_loader, dev))
    clean = precompute_joint_cache(tk[1][0], tk[1][1], train_ds, torch.zeros(Cb, device=dev),
                                   torch.ones(Cb, dtype=torch.bool, device=dev), dev, seed=7)
    R["floor"] = {"mean": float(np.mean(floor_d)), "dices": floor_d}
    R["ceiling"] = {"mean": float(np.mean(students(clean)))}

    save_results("final")

    # figure: headline uniform series + the best Joint K=10 keep2% for reference
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 6.5))
    plot_series = {
        "K=1 uniform": R["series"]["K=1 uniform"],
        "PATE K=3 uniform": R["series"]["PATE K=3 uniform"],
        "PATE K=5 uniform": R["series"]["PATE K=5 uniform"],
        "PATE K=10 uniform": R["series"]["PATE K=10 uniform"],
        "subsample K=1 keep2%": R["series"]["subsample K=1"]["keep2%"],
        "Joint K=10 keep2%": R["series"]["Joint K=10"]["keep2%"],
        "PATE K=5 + CANAL": R["series"].get("PATE K=5 + CANAL"),
    }
    plot_series = {k: v for k, v in plot_series.items() if v is not None}
    names = list(plot_series); x = np.arange(len(EPS)); nb = len(names); w = 0.8 / nb
    for i, n in enumerate(names):
        ms = [plot_series[n][f"{e}"]["mean"] for e in EPS]; ss = [plot_series[n][f"{e}"]["sem"] for e in EPS]
        hatch = "xx" if "CANAL" in n else ("//" if "K=1 uniform" in n else "")
        ax.bar(x + (i - nb / 2 + 0.5) * w, ms, w, yerr=ss, capsize=2, edgecolor="black",
               linewidth=1.1 if "Joint" in n else 0.7, hatch=hatch, label=n)
    ax.axhline(R["floor"]["mean"], color="black", ls="--", lw=1.5, label=f"floor {R['floor']['mean']:.3f}")
    ax.axhline(R["ceiling"]["mean"], color="green", ls=":", lw=1.7, label=f"ceiling {R['ceiling']['mean']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"ε={int(e)}" for e in EPS], fontsize=12)
    ax.set_ylabel("Dice (mean ± SEM)")
    ax.set_title(f"{args.dataset.upper()} comparison (n={len(train_ds)}, {args.seeds} seeds, K∈{K_VALUES})",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(HERE / f"fig_{args.dataset}_comparison.png", dpi=130)

    print(f"\nsaved results/{args.dataset}_comparison.json + fig_{args.dataset}_comparison.png")
    print(f"floor={R['floor']['mean']:.3f} ceiling={R['ceiling']['mean']:.3f}")


if __name__ == "__main__":
    main()