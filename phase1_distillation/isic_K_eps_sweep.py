# -*- coding: utf-8 -*-
"""
ISIC K-sweep x eps-sweep x {PATE+uniform, PATE+CANAL} comparison.

For each K in {1, 3, 5, 10}:
  - Train K teachers on disjoint cohorts
  - Compute shared act_norm importance (mean over K teachers)
  - For each eps in {0.1, 0.5, 1, 2, 4, 6, 8}:
    - PATE+uniform: sigma_unif from rho budget, all channels same sigma
    - PATE+CANAL : sigma_c = kappa sqrt(Delta) / s_c^(1/4)
  - SEEDS student trainings per cell

K=1 PATE+uniform is the "PATE single-teacher baseline" (no aggregation).
K=1 PATE+CANAL  is the original CANAL paper single-teacher water-filling.

Output:
  results/isic_K_eps_sweep{tag}_results.json
  fig_isic_K_eps_sweep{tag}.png

Usage:
  python isic_K_eps_sweep.py
  python isic_K_eps_sweep.py --Ks 1,3,5,10 --epsilons 0.1,0.5,1,2,4,6,8 --seeds 5
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from isic_dataset import ISICDataset
from drive_local_demo import (
    TinyUNet, evaluate_vessel_dice, compute_importance_actnorm, collect_caps,
)
from drive_pate_poc import train_K_teachers, correct_uniform_sigma, precompute_pate_cache
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho


HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def cleanup(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def shared_actnorm_importance(teachers, loader, device):
    """Average per-teacher act_norm importance into one shared C-vector."""
    imps = [compute_importance_actnorm(t, loader, device) for t in teachers]
    return torch.stack(imps, dim=0).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--Ks", type=str, default="1,3,5,10")
    ap.add_argument("--methods", type=str, default="uniform,CANAL",
                    help="comma-separated subset of {uniform,CANAL}")
    ap.add_argument("--epsilons", type=str, default="0.1,0.5,1,2,4,6,8")
    ap.add_argument("--te", type=int, default=50, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    ap.add_argument("--out_tag", type=str, default="",
                    help="suffix for output JSON/PNG to avoid overwriting prior runs")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    K_LIST = [int(k) for k in args.Ks.split(",")]
    METHODS = [m.strip() for m in args.methods.split(",")]
    assert all(m in ("uniform", "CANAL") for m in METHODS), f"bad --methods: {METHODS}"
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds = ISICDataset(split="train", size=args.size)
    val_ds = ISICDataset(split="val", size=args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[ISIC K-sweep] device={device} train={len(train_ds)} val={len(val_ds)}")
    print(f"  Ks={K_LIST}  EPS={EPS}  SEEDS={SEEDS}")

    train_loader_imp = DataLoader(train_ds, batch_size=8, shuffle=False)

    # ---- train teachers for each K, compute shared importance ----
    teachers_by_K = {}
    caps_by_K = {}
    imp_by_K = {}
    for K in K_LIST:
        print(f"\n[teachers] training K={K}")
        t0 = time.time()
        teachers, caps = train_K_teachers(
            train_ds, K=K, device=device, n_epochs=args.te, in_ch=3,
        )
        teachers_by_K[K] = teachers
        caps_by_K[K] = caps
        if K == 1:
            imp_by_K[K] = compute_importance_actnorm(
                teachers[0], train_loader_imp, device,
            ).to(device)
        else:
            imp_by_K[K] = shared_actnorm_importance(
                teachers, train_loader_imp, device,
            ).to(device)
        R = (imp_by_K[K].max() / imp_by_K[K].min()).item()
        print(f"  K={K} done in {time.time()-t0:.0f}s, R(act_norm)={R:.2f}")
        cleanup(device)

    Cb = teachers_by_K[K_LIST[0]][0].base * 4
    print(f"\nbottleneck channels C = {Cb}")

    # ---- build results scaffold ----
    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    save_path = HERE / "results" / f"isic_K_eps_sweep{tag_suffix}_results.json"
    results = {
        "dataset": "isic",
        "te": args.te, "se": args.se,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "Ks": K_LIST, "epsilons": EPS, "seeds": SEEDS,
        "importance_metric": "act_norm",
        "R_by_K": {
            str(K): float((imp_by_K[K].max() / imp_by_K[K].min()).item())
            for K in K_LIST
        },
        "C": Cb,
        "series": {},
    }

    def run_cell(teachers, caps_list, sigma, label, eps):
        cache = precompute_pate_cache(
            teachers, caps_list, train_ds, sigma, device,
            seed=42 + int(eps * 10),
        )
        dices = []
        for s in SEEDS:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=16, teacher_base=32,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4,
                seed=s, in_ch=3,
            )
            dices.append(best)
            print(f"      {label} eps={eps} seed={s}: {best:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            cleanup(device)
        del cache
        cleanup(device)
        return {
            "dices": dices,
            "mean": float(np.mean(dices)),
            "std": float(np.std(dices)),
            "sem": float(np.std(dices) / np.sqrt(len(dices))),
        }

    # ---- sweep K x eps x method ----
    total = len(K_LIST) * len(EPS) * len(METHODS)
    done = 0
    job_t0 = time.time()
    for K in K_LIST:
        deltas = torch.full((Cb,), 2.0 / K, device=device)
        for method in METHODS:
            series_key = f"PATE+{method} (K={K})"
            results["series"][series_key] = {}
            for eps in EPS:
                rho = eps_to_rho(eps)
                if method == "uniform":
                    sigma = correct_uniform_sigma(deltas, rho)
                    print(f"\n  [{series_key}] eps={eps}  sigma={sigma[0].item():.4f}")
                else:
                    sigma = correct_waterfilling_sigma(deltas, imp_by_K[K], rho)
                    top10 = torch.argsort(imp_by_K[K], descending=True)[:10]
                    bot10 = torch.argsort(imp_by_K[K], descending=False)[:10]
                    ratio = (sigma[bot10].mean() / sigma[top10].mean()).item()
                    print(f"\n  [{series_key}] eps={eps}  "
                          f"sigma top10={sigma[top10].mean().item():.4f}  "
                          f"bot10={sigma[bot10].mean().item():.4f}  ratio={ratio:.3f}x")
                results["series"][series_key][str(eps)] = run_cell(
                    teachers_by_K[K], caps_by_K[K], sigma,
                    f"K{K}-{method}", eps,
                )
                done += 1
                elapsed = time.time() - job_t0
                eta = elapsed / done * (total - done)
                print(f"  [progress] {done}/{total}  "
                      f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
                save_path.write_text(json.dumps(results, indent=2))

    # ---- final summary ----
    print("\n" + "=" * 110)
    print(f"ISIC K x eps sweep summary  (act_norm importance, {args.seeds} seeds)")
    print("=" * 110)
    hdr = f"{'series':<26s} " + "  ".join(f"{e:>7.2f}" for e in EPS)
    print(hdr)
    print("-" * len(hdr))
    for key, data in results["series"].items():
        row = "  ".join(f"{data[str(e)]['mean']:>7.4f}" for e in EPS)
        print(f"{key:<26s} {row}")
    print("=" * 110)
    print(f"R_by_K = {results['R_by_K']}")

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {save_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 7))
        colors_K = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e", 10: "#d62728"}
        styles_m = {"uniform": "-", "CANAL": "--"}
        markers_m = {"uniform": "o", "CANAL": "^"}
        for K in K_LIST:
            for method in METHODS:
                key = f"PATE+{method} (K={K})"
                d = results["series"][key]
                means = [d[str(e)]["mean"] for e in EPS]
                stds = [d[str(e)]["std"] for e in EPS]
                ax.errorbar(
                    EPS, means, yerr=stds,
                    color=colors_K[K], ls=styles_m[method],
                    marker=markers_m[method],
                    lw=2, ms=8, capsize=4,
                    label=f"K={K} {method}",
                )
        ax.set_xscale("log")
        ax.set_xticks(EPS)
        ax.set_xticklabels([str(e) for e in EPS])
        ax.set_xlabel("privacy budget  eps (user-level, sample-once)")
        ax.set_ylabel("Dice (ISIC val)")
        ax.set_title(
            f"ISIC K-sweep x eps-sweep  "
            f"(K in {K_LIST}, {args.seeds} seeds, act_norm)"
        )
        ax.legend(loc="best", ncol=4, fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = HERE / f"fig_isic_K_eps_sweep{tag_suffix}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    cleanup(device)


if __name__ == "__main__":
    main()
