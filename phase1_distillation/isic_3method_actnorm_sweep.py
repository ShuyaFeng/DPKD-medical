"""
ISIC 3-method sweep with the NEW act_norm importance (commit e37ec36).

Methods compared (5 seeds × 7 eps each):
  1. PATE           — K=1 baseline (single teacher, uniform sigma)
  2. PATE + uniform — K=10 PATE aggregation, uniform sigma on aggregate
  3. PATE + CANAL   — K=10 PATE aggregation, waterfilling sigma using
                      shared act_norm importance (NEW importance metric)

epsilons = {0.1, 0.5, 1, 2, 4, 6, 8}  (user-specified)
Cleans up MPS / CUDA cache between cells to keep memory bounded.
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
from drive_pate_poc import (
    train_K_teachers, correct_uniform_sigma, precompute_pate_cache,
)
from drive_pate_canal_combined import correct_waterfilling_sigma
from drive_student_distill import train_student_distill
from synthetic_demo import eps_to_rho


HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)


def cleanup_memory(device: str):
    """Release GPU / MPS cache and trigger Python GC."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def shared_actnorm_importance(teachers, train_loader, device):
    """Average K teachers' act_norm importance into ONE shared channel ranking.

    Uses the NEW compute_importance_actnorm (commit e37ec36), not
    grad_energy. No backward pass needed -> faster and more memory-friendly.
    """
    imps = [compute_importance_actnorm(t, train_loader, device) for t in teachers]
    return torch.stack(imps, dim=0).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", type=str,
                    default="0.1,0.5,1,2,4,6,8")
    ap.add_argument("--te", type=int, default=50, help="teacher epochs")
    ap.add_argument("--se", type=int, default=40, help="student epochs")
    ap.add_argument("--out_tag", type=str, default="",
                    help="suffix for output JSON/PNG to avoid overwriting prior runs")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    EPS = [float(e) for e in args.epsilons.split(",")]
    SEEDS = list(range(100, 100 + args.seeds * 100, 100))

    train_ds = ISICDataset(split="train", size=args.size)
    val_ds   = ISICDataset(split="val",   size=args.size)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"[ISIC] device={device}  train={len(train_ds)} val={len(val_ds)}  "
          f"eps={EPS}  seeds={SEEDS}")

    # ----------------------------------------------------------------
    # Teachers — K=1 (baseline) and K=10 (PATE)
    # ----------------------------------------------------------------
    print("\n[teachers] training K=1 (baseline)...")
    t1_start = time.time()
    teachers_k1, caps_k1 = train_K_teachers(train_ds, K=1, device=device,
                                            n_epochs=args.te, in_ch=3)
    cleanup_memory(device)
    print(f"  K=1 trained in {time.time() - t1_start:.1f}s")

    print("\n[teachers] training K=10 (PATE)...")
    t10_start = time.time()
    teachers_k10, caps_k10 = train_K_teachers(train_ds, K=10, device=device,
                                              n_epochs=args.te, in_ch=3)
    cleanup_memory(device)
    print(f"  K=10 trained in {time.time() - t10_start:.1f}s")

    Cb = teachers_k1[0].base * 4
    print(f"  bottleneck channels C = {Cb}")

    # ----------------------------------------------------------------
    # Shared importance for K=10 PATE+CANAL  (NEW act_norm metric)
    # ----------------------------------------------------------------
    print("\n[importance] computing shared act_norm importance over K=10...")
    train_loader_imp = DataLoader(train_ds, batch_size=8, shuffle=False)
    imp_k10 = shared_actnorm_importance(teachers_k10, train_loader_imp, device).to(device)
    print(f"  act_norm importance: min={imp_k10.min():.4e}  "
          f"max={imp_k10.max():.4e}  ratio={(imp_k10.max()/imp_k10.min()).item():.2f}x")
    cleanup_memory(device)

    # ----------------------------------------------------------------
    # Sweep
    # ----------------------------------------------------------------
    results = {
        "dataset": "isic",
        "n_train": len(train_ds), "n_val": len(val_ds),
        "epsilons": EPS, "seeds": SEEDS,
        "importance_metric": "act_norm (new, commit e37ec36)",
        "importance_ratio_K10": float(imp_k10.max() / imp_k10.min()),
        "series": {
            "PATE (K=1 baseline)":    {},
            "PATE+uniform (K=10)":    {},
            "PATE+CANAL (K=10, act_norm)": {},
        },
    }

    deltas_k1  = torch.full((Cb,), 2.0,        device=device)  # K=1: Delta=2
    deltas_k10 = torch.full((Cb,), 2.0 / 10.0, device=device)  # K=10: Delta=2/10

    def run_cell(teachers, caps_list, sigma, label, eps):
        cache = precompute_pate_cache(teachers, caps_list, train_ds, sigma,
                                      device, seed=42 + int(eps * 10))
        dices = []
        for s in SEEDS:
            t0 = time.time()
            best, _ = train_student_distill(
                train_ds, val_loader, cache, device,
                student_base=16, teacher_base=32,
                n_epochs=args.se, lr=1e-3, lambda_feat=0.4, seed=s, in_ch=3,
            )
            dices.append(best)
            print(f"      {label} eps={eps} seed={s}: {best:.4f}  ({time.time() - t0:.1f}s)")
            cleanup_memory(device)
        del cache
        cleanup_memory(device)
        return {
            "dices": dices,
            "mean": float(np.mean(dices)),
            "std":  float(np.std(dices)),
            "sem":  float(np.std(dices) / np.sqrt(len(dices))),
        }

    tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    save_path = HERE / "results" / f"isic_3method_actnorm{tag_suffix}_sweep_results.json"

    total_cells = 3 * len(EPS)
    cell_done = 0
    job_start = time.time()

    for eps in EPS:
        rho = eps_to_rho(eps)
        print(f"\n========== eps = {eps}  (rho = {rho:.4f}) ==========")

        # ----- Method 1: PATE (K=1 baseline, uniform) -----
        sigma = correct_uniform_sigma(deltas_k1, rho)
        print(f"   [PATE K=1]      sigma = {sigma[0].item():.4f}")
        results["series"]["PATE (K=1 baseline)"][str(eps)] = run_cell(
            teachers_k1, caps_k1, sigma, "PATE-K1", eps,
        )
        cell_done += 1
        elapsed = time.time() - job_start
        eta = elapsed / cell_done * (total_cells - cell_done)
        print(f"   [progress] {cell_done}/{total_cells} cells  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

        # ----- Method 2: PATE + uniform (K=10) -----
        sigma = correct_uniform_sigma(deltas_k10, rho)
        print(f"   [PATE K=10 unif] sigma = {sigma[0].item():.4f}")
        results["series"]["PATE+uniform (K=10)"][str(eps)] = run_cell(
            teachers_k10, caps_k10, sigma, "PATE-K10-uni", eps,
        )
        cell_done += 1
        elapsed = time.time() - job_start
        eta = elapsed / cell_done * (total_cells - cell_done)
        print(f"   [progress] {cell_done}/{total_cells} cells  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

        # ----- Method 3: PATE + CANAL (K=10, act_norm importance) -----
        sigma = correct_waterfilling_sigma(deltas_k10, imp_k10, rho)
        top10 = torch.argsort(imp_k10, descending=True)[:10]
        bot10 = torch.argsort(imp_k10, descending=False)[:10]
        ratio = (sigma[bot10].mean() / sigma[top10].mean()).item()
        print(f"   [PATE K=10 CANAL] sigma top10={sigma[top10].mean().item():.4f}  "
              f"bot10={sigma[bot10].mean().item():.4f}  ratio={ratio:.3f}x")
        results["series"]["PATE+CANAL (K=10, act_norm)"][str(eps)] = run_cell(
            teachers_k10, caps_k10, sigma, "PATE-K10-CANAL", eps,
        )
        cell_done += 1
        elapsed = time.time() - job_start
        eta = elapsed / cell_done * (total_cells - cell_done)
        print(f"   [progress] {cell_done}/{total_cells} cells  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
        save_path.write_text(json.dumps(results, indent=2))

    # ----------------------------------------------------------------
    # Final summary + plot
    # ----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("ISIC 3-method sweep — act_norm importance — final summary")
    print("=" * 92)
    hdr = f"{'eps':>6}  " + "  ".join(f"{name[:24]:>24s}" for name in results["series"])
    print(hdr)
    print("-" * len(hdr))
    for eps in EPS:
        row = f"{eps:>6.2f}  "
        for name in results["series"]:
            m = results["series"][name][str(eps)]
            row += f"{m['mean']:>10.4f} ± {m['std']:.4f}    "
        print(row)
    print("=" * 92)

    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {save_path}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        colors = {"PATE (K=1 baseline)":         "#d62728",
                  "PATE+uniform (K=10)":         "#2ca02c",
                  "PATE+CANAL (K=10, act_norm)": "#9467bd"}
        markers = {"PATE (K=1 baseline)": "s",
                   "PATE+uniform (K=10)": "o",
                   "PATE+CANAL (K=10, act_norm)": "^"}
        for name, data in results["series"].items():
            means = [data[str(e)]["mean"] for e in EPS]
            stds  = [data[str(e)]["std"]  for e in EPS]
            ax.errorbar(EPS, means, yerr=stds, fmt=f"{markers[name]}-",
                        color=colors[name], lw=2.2, ms=10, capsize=6,
                        label=name)
        ax.set_xscale("log")
        ax.set_xticks(EPS)
        ax.set_xticklabels([str(e) for e in EPS])
        ax.set_xlabel("privacy budget  eps (user-level, sample-once)")
        ax.set_ylabel("lesion Dice (ISIC val)")
        ax.set_title(f"ISIC PATE / PATE+uniform / PATE+CANAL  ({args.seeds} seeds; "
                     f"new act_norm importance)")
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = HERE / f"fig_isic_3method_actnorm{tag_suffix}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    cleanup_memory(device)


if __name__ == "__main__":
    main()
